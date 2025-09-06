import os
import time
import hashlib
import json
import logging
import random
import shutil
import concurrent.futures
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type, TypeVar, overload, ClassVar

import diskcache
from tqdm import tqdm

from openai import OpenAI
import anthropic
from google import genai
from pydantic import BaseModel as ResponseSchema, ValidationError

# Use a dedicated logger for this module for better isolation in larger apps.
logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "my_llms"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache = diskcache.Cache(str(CACHE_DIR))

RS = TypeVar("RS", bound=ResponseSchema)

def clear_cache():
    """Clear the entire LLM response cache."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache at {CACHE_DIR}")
    else:
        print(f"No cache found at {CACHE_DIR}")

def get_cache_size() -> float:
    """Get the total cache size in megabytes."""
    if not CACHE_DIR.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in CACHE_DIR.rglob('*') if f.is_file())
    return round(total_bytes / (1024 * 1024), 2)

def list_cache():
    """Print information about the cache."""
    if not CACHE_DIR.exists():
        print("No cache found")
        return
    size_mb = get_cache_size()
    print(f"my_llms_refactored cache: {size_mb:.2f} MB at {CACHE_DIR}")

def get_cache_key(model_name: str, prompt: str, **kwargs) -> str:
    """Create a deterministic cache key from all relevant parameters."""
    # Operate on a copy to avoid side effects on the caller's dictionary.
    kwargs_copy = kwargs.copy()
    if 'response_schema' in kwargs_copy and kwargs_copy['response_schema']:
        kwargs_copy['response_schema'] = kwargs_copy['response_schema'].model_json_schema()

    serializable_kwargs = {k: str(v) for k, v in kwargs_copy.items()}
    sorted_kwargs = json.dumps(serializable_kwargs, sort_keys=True)
    key_string = f"{model_name}::{prompt}::{sorted_kwargs}"
    return hashlib.sha256(key_string.encode('utf-8')).hexdigest()


class MyApiModel(ABC):
    """
    Abstract base class for LLM API models.

    Handles caching, retries, and batch processing. Subclasses must implement
    `_initialize_client` and `_perform_api_call`.
    """
    supports_response_schema: ClassVar[bool] = False

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name")
        if not self.model_name:
            raise ValueError("'model_name' is required in the configuration.")

        self.generation_config = config.get("generation_config", {})
        self.num_tries_per_request = config.get("num_tries_per_request", 3)
        self.use_cache = config.get("use_cache", True)
        self.verbose = config.get("verbose", False)

        self.client = self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize and return the provider-specific API client."""
        raise NotImplementedError

    def _get_api_key(self, env_var: str) -> str:
        """Retrieve API key from config or environment variable."""
        api_key = self.config.get("api_key") or os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"API key not found. Please provide it via 'api_key' in "
                f"the config or set the {env_var} environment variable."
            )
        return api_key

    def _filter_params(self, params: dict[str, Any], supported: set[str]) -> dict[str, Any]:
        """Filter a dictionary of parameters against a set of supported ones."""
        return {k: v for k, v in params.items() if k in supported}

    @overload
    def __call__(self, prompts: str, **kwargs) -> str: ...
    @overload
    def __call__(self, prompts: list[str], **kwargs) -> list[str | Exception]: ...
    @overload
    def __call__(self, prompts: str, *, response_schema: Type[RS], **kwargs) -> RS: ...
    @overload
    def __call__(self, prompts: list[str], *, response_schema: Type[RS], **kwargs) -> list[RS | Exception]: ...

    def __call__(self, prompts, **kwargs):
        """
        Generate responses for one or more prompts with concurrency.

        For batch calls, this method is resilient to individual failures. It returns
        a list containing results for successful calls and Exception objects for
        any failed calls, preserving the original order. The caller is responsible
        for handling any exceptions in the returned list.
        """
        if not prompts:
            raise ValueError("Prompts cannot be empty.")

        is_single = isinstance(prompts, str)
        prompt_list = [prompts] if is_single else prompts

        if not all(isinstance(p, str) and p.strip() for p in prompt_list):
            raise ValueError("All prompts must be non-empty strings.")

        if is_single:
            return self.one_call(prompt_list[0], **kwargs)

        max_workers = min(os.cpu_count() or 1, 16)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {executor.submit(self.one_call, p, **kwargs): p for p in prompt_list}
            prompt_to_result = {}

            iterator = concurrent.futures.as_completed(future_to_prompt)
            if self.verbose:
                iterator = tqdm(iterator, desc=f"Processing {len(prompt_list)} prompts", total=len(prompt_list))

            for future in iterator:
                prompt = future_to_prompt[future]
                try:
                    prompt_to_result[prompt] = future.result()
                except Exception as e:
                    prompt_to_result[prompt] = e
            
            return [prompt_to_result[p] for p in prompt_list]

    def one_call(self, prompt: str, **kwargs) -> str | ResponseSchema:
        """Orchestrate a single API call with caching and retries."""
        generation_args = self.generation_config.copy()
        generation_args.update(kwargs)
        response_schema = generation_args.get("response_schema")

        if response_schema and not self.supports_response_schema:
            raise NotImplementedError(f"Model '{self.model_name}' does not support response schemas.")

        cache_key = None
        if self.use_cache:
            cache_key = get_cache_key(self.model_name, prompt, **generation_args)
            cached_item = cache.get(cache_key)
            if cached_item:
                if response_schema:
                    try:
                        return self._parse_and_validate_json(cached_item, response_schema)
                    except (json.JSONDecodeError, ValidationError):
                        pass
                elif isinstance(cached_item, str):
                    return cached_item

        last_exception = None
        for attempt in range(self.num_tries_per_request):
            try:
                content = self._perform_api_call(prompt, **generation_args)

                if not content or not content.strip():
                    raise ValueError("API returned an empty response.")

                if response_schema:
                    response_data = self._parse_and_validate_json(content, response_schema)
                else:
                    response_data = content

                if self.use_cache and cache_key:
                    item_to_cache = response_data.model_dump_json() if response_schema else response_data
                    cache.set(cache_key, item_to_cache)

                return response_data

            except Exception as e:
                last_exception = e
                if self.verbose:
                    logger.warning(f"Attempt {attempt + 1} failed for {self.model_name}: {e}")
                if attempt < self.num_tries_per_request - 1:
                    # Exponential backoff with jitter to prevent thundering herd.
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)

        raise RuntimeError(
            f"API call failed after {self.num_tries_per_request} attempts. "
            f"Last error: {last_exception}"
        ) from last_exception

    def _parse_and_validate_json(self, content: str, schema: Type[RS]) -> RS:
        """Helper to decode and validate JSON against a Pydantic schema."""
        try:
            data = json.loads(content)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to validate response against schema: {e}\nRaw content: {content}")
            raise

    @abstractmethod
    def _perform_api_call(self, prompt: str, **generation_args) -> str:
        """
        Perform the provider-specific API call and extract text content.
        """
        raise NotImplementedError


class MyOpenAIModel(MyApiModel):
    """OpenAI API wrapper (e.g., GPT-4, GPT-3.5)."""
    supports_response_schema = True
    _supported_params = {
        "temperature", "max_tokens", "top_p", "n", "stop",
        "presence_penalty", "frequency_penalty", "logit_bias",
        "response_format", "seed", "tools", "tool_choice"
    }

    def _initialize_client(self) -> OpenAI:
        api_key = self._get_api_key("OPENAI_API_KEY")
        timeout = self.config.get("timeout", 60.0)
        return OpenAI(api_key=api_key, timeout=timeout)

    def _perform_api_call(self, prompt: str, **generation_args) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        if generation_args.get("response_schema"):
            if "json" not in prompt.lower():
                messages[0]["content"] += "\n\nPlease respond in valid JSON format."
            generation_args["response_format"] = {"type": "json_object"}
        
        api_args = self._filter_params(generation_args, self._supported_params)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **api_args
        )

        content = response.choices[0].message.content
        return content.strip() if content else ""


class MyGoogleModel(MyApiModel):
    """Google Gemini API wrapper."""
    supports_response_schema = True
    _supported_params = {"temperature", "max_output_tokens", "top_p", "top_k", "stop_sequences"}

    def _initialize_client(self) -> genai.Client:
        api_key = self._get_api_key("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key)

    def _perform_api_call(self, prompt: str, **generation_args) -> str:
        if "max_tokens" in generation_args:
            generation_args["max_output_tokens"] = generation_args.pop("max_tokens")
        
        gen_config = self._filter_params(generation_args, self._supported_params)

        if response_schema := generation_args.get("response_schema"):
            gen_config["response_schema"] = response_schema.model_json_schema()
            gen_config["response_mime_type"] = "application/json"
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=gen_config
        )

        if response.text:
            return response.text.strip()
        
        logger.warning("Gemini response was blocked or empty.")
        return ""


class MyAnthropicModel(MyApiModel):
    """Anthropic Claude API wrapper."""
    supports_response_schema = False
    _supported_params = {"max_tokens", "temperature", "top_p", "top_k", "stop_sequences", "metadata"}

    def _initialize_client(self) -> anthropic.Anthropic:
        api_key = self._get_api_key("ANTHROPIC_API_KEY")
        timeout = self.config.get("timeout", 60.0)
        return anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def _perform_api_call(self, prompt: str, **generation_args) -> str:
        api_args = self._filter_params(generation_args, self._supported_params)
        if "max_tokens" not in api_args:
            api_args["max_tokens"] = 2048

        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **api_args
        )

        if response.content and isinstance(response.content[0], anthropic.types.TextBlock):
            return response.content[0].text.strip()
        return ""


