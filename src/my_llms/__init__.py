"""Simple LLM API wrapper with caching."""

from .api_models import (
    MyApiModel, 
    MyOpenAIModel, 
    MyGoogleModel, 
    MyAnthropicModel,
    clear_cache,
    get_cache_size,
    list_cache,
    CACHE_DIR,
)

__version__ = "0.1.0"
__author__ = "Anton Xue"
__email__ = "anton.xue@example.com"


def load_model(config: dict):
    """Load model from provider config."""
    provider = config.get("provider", None)
    if provider == "openai":
        return MyOpenAIModel(config)
    elif provider == "anthropic":
        return MyAnthropicModel(config)
    elif provider == "google":
        return MyGoogleModel(config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


__all__ = [
    "MyApiModel",
    "MyOpenAIModel", 
    "MyGoogleModel",
    "MyAnthropicModel",
    "clear_cache",
    "get_cache_size",
    "list_cache",
    "CACHE_DIR",
    "load_model",
]