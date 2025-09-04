# my-llms

Simple Python wrapper for LLM APIs (OpenAI, Anthropic, Google) with caching and batch processing.

## Features

- **Multiple providers**: OpenAI, Anthropic, Google
- **Unified interface**: Same API across all providers  
- **Automatic caching**: Fast local response caching
- **Batch processing**: Concurrent request handling
- **Structured output**: Pydantic validation (OpenAI & Google)
- **Auto retries**: Built-in error handling

## Installation

```bash
# From PyPI
pip install my-llms

# For local development
pip install -e .

# With optional dependencies
pip install -e .[dev,test]
```

## Quick Start

```python
from my_llms import load_model

# Load any provider
model = load_model({"provider": "openai", "model_name": "gpt-4o-mini"})

# Single prompt
response = model("What is the capital of France?")
print(response)  # "The capital of France is Paris."

# Batch prompts (concurrent)
responses = model(["What is 2+2?", "What is 3+3?"])
print(responses)  # ["2+2 equals 4.", "3+3 equals 6."]
```

## Usage Examples

### Basic Usage with Different Providers

```python
from my_llms import load_model

# OpenAI
openai_model = load_model({
    "provider": "openai",
    "model_name": "gpt-4o-mini"
})

# Anthropic Claude
claude_model = load_model({
    "provider": "anthropic", 
    "model_name": "claude-3-haiku-20240307"
})

# Google Gemini
gemini_model = load_model({
    "provider": "google",
    "model_name": "gemini-2.0-flash-exp"
})

# All work the same way
question = "Explain quantum computing in one sentence."
print(openai_model(question))
print(claude_model(question))
print(gemini_model(question))
```

### Configuration Options

```python
model = load_model({
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "generation_config": {
        "temperature": 0.7,      # Controls creativity (0.0-2.0)
        "max_tokens": 150,       # Response length limit
        "top_p": 0.9            # Nucleus sampling
    },
    "use_cache": True,           # Enable caching (default: True)
    "num_tries_per_request": 3,  # Retry attempts (default: 3)
    "verbose": True              # Show progress bars (default: False)
})
```

### API Keys Setup

```python
# Method 1: Environment variables (recommended)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  
os.environ["GOOGLE_API_KEY"] = "AIza..."

# Method 2: Pass in config
model = load_model({
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "api_key": "sk-your-key-here"
})
```

### Batch Processing

```python
model = load_model({
    "provider": "openai", 
    "model_name": "gpt-4o-mini",
    "verbose": True  # Show progress bar
})

# Process many prompts concurrently
prompts = [
    "Translate 'hello' to Spanish",
    "Translate 'goodbye' to French", 
    "Translate 'thank you' to German",
    "Translate 'please' to Italian",
    "Translate 'yes' to Portuguese"
]

responses = model(prompts)
for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

### Structured Output with Pydantic (OpenAI & Google)

```python
from my_llms import load_model
from pydantic import BaseModel

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    prep_time_minutes: int
    difficulty: str  # "easy", "medium", "hard"

model = load_model({"provider": "openai", "model_name": "gpt-4o-mini"})

recipe = model(
    "Give me a simple pasta recipe",
    response_schema=Recipe
)

print(f"Recipe: {recipe.name}")
print(f"Ingredients: {', '.join(recipe.ingredients)}")
print(f"Prep time: {recipe.prep_time_minutes} minutes")
print(f"Difficulty: {recipe.difficulty}")
```

### Dynamic Parameters

```python
model = load_model({"provider": "openai", "model_name": "gpt-4o-mini"})

# Override config per call
creative_response = model(
    "Write a creative story about a robot",
    temperature=1.2,  # More creative
    max_tokens=200
)

factual_response = model(
    "What is the population of Tokyo?",
    temperature=0.1,  # More factual
    max_tokens=50
)
```

### Cache Management

```python
from my_llms import clear_cache, get_cache_size, list_cache

# Check cache status
print(f"Cache size: {get_cache_size()} MB")

# List cache info
list_cache()

# Clear cache when needed
clear_cache()

# Disable cache for specific model
no_cache_model = load_model({
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "use_cache": False
})
```

### Error Handling

```python
from my_llms import load_model

try:
    model = load_model({
        "provider": "openai",
        "model_name": "gpt-4o-mini"
        # Missing API key
    })
    response = model("Hello")
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"API error: {e}")

# Handle batch failures gracefully
model = load_model({
    "provider": "openai", 
    "model_name": "gpt-4o-mini",
    "num_tries_per_request": 5,  # More retries
    "verbose": True  # See retry attempts
})
```

### Working with Different Model Types

```python
# Code generation
code_model = load_model({
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "generation_config": {"temperature": 0.2}  # More deterministic
})

code = code_model("Write a Python function to calculate fibonacci numbers")

# Creative writing  
creative_model = load_model({
    "provider": "anthropic",
    "model_name": "claude-3-haiku-20240307",
    "generation_config": {"temperature": 0.9}  # More creative
})

story = creative_model("Write a short sci-fi story")

# Analysis tasks
analytical_model = load_model({
    "provider": "google", 
    "model_name": "gemini-2.0-flash-exp",
    "generation_config": {"temperature": 0.1}  # Very factual
})

analysis = analytical_model("Analyze the pros and cons of renewable energy")
```

### Advanced Batch Processing with Mixed Parameters

```python
model = load_model({"provider": "openai", "model_name": "gpt-4o-mini"})

# Different temperature for each type of question
questions = [
    "What is 2+2?",                    # Factual
    "Write a creative poem",           # Creative  
    "Explain photosynthesis",          # Educational
    "Tell me a joke"                   # Creative
]

# Process with different parameters
responses = []
for i, question in enumerate(questions):
    temp = 0.1 if "what" in question.lower() or "explain" in question.lower() else 0.8
    response = model(question, temperature=temp)
    responses.append(response)
```

## License

MIT License - see LICENSE file for details.