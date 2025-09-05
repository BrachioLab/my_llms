import os
import time
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
import concurrent.futures
from abc import abstractmethod
import shutil
from tqdm import tqdm
import diskcache
from openai import OpenAI
import anthropic
from google import genai
from pydantic import BaseModel as ResponseSchema, ValidationError

# Cache setup
CACHE_DIR = Path.home() / ".cache" / "my_llms"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache = diskcache.Cache(str(CACHE_DIR))


def clear_cache():
    """Clear cache."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache at {CACHE_DIR}")
    else:
        print(f"No cache found at {CACHE_DIR}")


def get_cache_size():
    """Get cache size in MB."""
    if not CACHE_DIR.exists():
        return 0.0
    total = sum(f.stat().st_size for f in CACHE_DIR.rglob('*') if f.is_file())
    return round(total / (1024 * 1024), 2)


def list_cache():
    """Show cache info."""
    if not CACHE_DIR.exists():
        print("No cache found")
        return
    
    size_mb = get_cache_size()
    print(f"my_llms cache: {size_mb:.2f} MB at {CACHE_DIR}")


def get_cache_key(model_name: str, prompt: str, **kwargs) -> str:
    """Create cache key from parameters."""
    serializable_kwargs = {k: str(v) for k, v in kwargs.items()}
    sorted_kwargs = json.dumps(serializable_kwargs, sort_keys=True)
    key_string = f"{model_name}::{prompt}::{sorted_kwargs}"
    return hashlib.sha256(key_string.encode('utf-8')).hexdigest()


class MyApiModel:
    """Base API model with caching, retries, batch processing."""

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model_name")
        if not self.model_name:
            raise ValueError("model_name is required in config")

        self.generation_config = config.get("generation_config", {})
        self.num_tries_per_request = config.get("num_tries_per_request", 3)
        self.use_cache = config.get("use_cache", True)
        self.verbose = config.get("verbose", False)
        
        self.client = self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize API client."""
        raise NotImplementedError

    def __call__(self, prompts: str | list[str], **kwargs) -> str | ResponseSchema | list[str] | list[ResponseSchema]:
        """Generate responses for prompt(s)."""
        if not prompts:
            raise ValueError("Prompts cannot be empty")
        
        is_single = isinstance(prompts, str)
        prompt_list = [prompts] if is_single else prompts
        
        for i, prompt in enumerate(prompt_list):
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Prompt {i} must be a non-empty string")
        
        if is_single:
            return self.one_call(prompt_list[0], **kwargs)
        
        max_workers = min(os.cpu_count() or 1, 24)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.one_call, p, **kwargs) for p in prompt_list]
            if self.verbose:
                futures = tqdm(futures, desc=f"Processing {len(prompt_list)} prompts")
            return [f.result() for f in futures]


    def one_call(self, prompt: str, **kwargs) -> str | ResponseSchema:
        """Handle single API call with caching and retries."""
        generation_args = self.generation_config.copy()
        generation_args.update(kwargs)
        
        response_schema = generation_args.get("response_schema")

        if self.use_cache:
            cache_key = get_cache_key(self.model_name, prompt, **generation_args)
            cached_item = cache.get(cache_key)
            if cached_item:
                if response_schema:
                    try:
                        if isinstance(cached_item, str):
                            return response_schema.model_validate(json.loads(cached_item))
                        elif isinstance(cached_item, dict):
                            return response_schema.model_validate(cached_item)
                    except (json.JSONDecodeError, ValidationError):
                        pass
                elif isinstance(cached_item, str):
                    return cached_item

        payload = self.prompt_to_payload(prompt)
        response_data = None
        last_exception = None

        for attempt in range(self.num_tries_per_request):
            try:
                response_data = self.get_response(payload, **generation_args)
                if response_data and (
                    isinstance(response_data, ResponseSchema)
                    or (
                        isinstance(response_data, str)
                        and len(response_data.strip()) > 0
                    )
                ):
                    if self.use_cache:
                        cache_key = get_cache_key(self.model_name, prompt, **generation_args)
                        if isinstance(response_data, ResponseSchema):
                            item_to_cache = response_data.model_dump_json()
                        else:
                            item_to_cache = response_data
                        cache.set(cache_key, item_to_cache)
                    break
            except Exception as e:
                last_exception = e
                if self.verbose:
                    print(f"Attempt {attempt + 1} failed for {self.model_name}: {e}")
                if attempt < self.num_tries_per_request - 1:
                    time.sleep(5 * (2 ** attempt))

        if response_data is None:
            raise RuntimeError(
                f"API call failed for {self.model_name} after {self.num_tries_per_request} attempts. "
                f"Last error: {last_exception}"
            )

        return response_data or ("" if not response_schema else None)

    @abstractmethod
    def prompt_to_payload(self, prompt: str) -> Any:
        """Convert prompt to API payload."""
        raise NotImplementedError

    @abstractmethod
    def get_response(self, payload: Any, **kwargs) -> str | ResponseSchema:
        """Make API call and return response."""
        raise NotImplementedError


class MyOpenAIModel(MyApiModel):
    """OpenAI API wrapper."""
    def _initialize_client(self) -> OpenAI:
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )
        
        timeout = self.config.get("timeout", 60.0)
        return OpenAI(api_key=api_key, timeout=timeout)

    def prompt_to_payload(self, prompt: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": prompt}]

    def get_response(self, payload: list[dict[str, Any]], **generation_args) -> str | ResponseSchema:
        response_schema = generation_args.pop("response_schema", None)
        if response_schema:
            generation_args["response_format"] = {"type": "json_object"}
            for i, msg in enumerate(payload):
                if "json" not in msg.get("content", "").lower():
                    payload[i]["content"] += "\nPlease respond in JSON format."
        
        supported_params = {
            "temperature", "max_tokens", "top_p", "n", "stop",
            "presence_penalty", "frequency_penalty", "logit_bias",
            "response_format", "seed", "tools", "tool_choice",
            "stream", "logprobs", "top_logprobs"
        }
        filtered_args = {k: v for k, v in generation_args.items() if k in supported_params}
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=payload,
                **filtered_args
            )
            content = response.choices[0].message.content
            if content is None: raise RuntimeError("Response content is None")

            if response_schema:
                try:
                    return response_schema.model_validate(json.loads(content))
                except (json.JSONDecodeError, ValidationError) as e:
                    logging.error(f"Failed to validate response: {e}")
                    logging.error(f"Raw content: {content}")
                    raise
            return content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e


class MyGoogleModel(MyApiModel):
    """Google Gemini API wrapper."""
    def _initialize_client(self) -> genai.Client:
        api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. "
                "Please provide it in the config or set the GOOGLE_API_KEY environment variable."
            )
        
        return genai.Client(api_key=api_key)

    def prompt_to_payload(self, prompt: str) -> str:
        return prompt

    def get_response(self, payload: str, **generation_args) -> str | ResponseSchema:
        response_schema = generation_args.pop("response_schema", None)
        if response_schema:
            generation_args["response_mime_type"] = "application/json"
            generation_args["response_schema"] = response_schema.model_json_schema()
        
        if "max_tokens" in generation_args:
            generation_args["max_output_tokens"] = generation_args.pop("max_tokens")
        
        supported_params = {
            "temperature", "max_output_tokens", "top_p", "top_k", 
            "response_mime_type", "response_schema", "stop_sequences"
        }
        filtered_args = {k: v for k, v in generation_args.items() if k in supported_params}

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=payload,
                config=filtered_args
            )
            content = response.text
            if content is None: raise RuntimeError("Response text is None")

            if response_schema:
                try:
                    return response_schema.model_validate(json.loads(content))
                except (json.JSONDecodeError, ValidationError) as e:
                    logging.error(f"Failed to validate response: {e}")
                    logging.error(f"Raw content: {content}")
                    raise
            return content.strip()
        except Exception as e:
            raise RuntimeError(f"Google API error: {e}") from e


class MyAnthropicModel(MyApiModel):
    """Anthropic Claude API wrapper."""
    def _initialize_client(self) -> anthropic.Anthropic:
        api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Please provide it in the config or set the ANTHROPIC_API_KEY environment variable."
            )
        
        timeout = self.config.get("timeout", 60.0)
        return anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def prompt_to_payload(self, prompt: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": prompt}]

    def get_response(self, payload: list[dict[str, Any]], **generation_args) -> str | ResponseSchema:
        generation_args.pop("response_schema", None)
        
        supported_params = {
            "max_tokens", "temperature", "top_p", "top_k",
            "stop_sequences", "stream", "metadata"
        }
        filtered_args = {k: v for k, v in generation_args.items() if k in supported_params}
        
        if "max_tokens" not in filtered_args:
            filtered_args["max_tokens"] = 1024
        
        try:
            response = self.client.messages.create(
                model=self.model_name, 
                messages=payload, 
                **filtered_args
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}") from e
