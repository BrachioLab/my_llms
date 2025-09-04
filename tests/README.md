# Tests for my_llms

This directory contains comprehensive tests for the my_llms package.

## Test Files

- `test_all.py` - comprehensive test suite with both unit and integration tests
- `conftest.py` - pytest configuration and fixtures

## Running Tests

### Prerequisites

1. Install the package in development mode:
   ```bash
   pip install -e .
   ```

2. Set up API keys (optional, tests will skip if not available):
   ```bash
   export OPENAI_API_KEY="your-key"
   export GOOGLE_API_KEY="your-key"  
   export ANTHROPIC_API_KEY="your-key"
   ```

### Run with pytest (recommended for unit tests)

```bash
# Run unit tests with pytest
pytest tests/test_all.py

# Run with verbose output
pytest -v tests/test_all.py

# Run specific test class
pytest tests/test_all.py::TestLoadModel

# Run specific test method
pytest tests/test_all.py::TestLoadModel::test_load_model_invalid_provider
```

### Run integration tests directly

```bash
# Run all integration tests
python tests/test_all.py

# Run specific test categories
python tests/test_all.py --test api
python tests/test_all.py --test cache
python tests/test_all.py --test error
python tests/test_all.py --test concurrent
python tests/test_all.py --test retry
python tests/test_all.py --test params

# Run with pytest (for unit tests only)
python tests/test_all.py --pytest
```

## Test Categories

### Unit Tests (pytest-compatible)
- **TestLoadModel**: Model creation and loading validation
- **TestAPIModelCreation**: API model instantiation tests
- **TestCaching**: Basic cache functionality tests  
- **TestModelBehavior**: Parameter validation and error handling

### Integration Tests (make real API calls)
- **API Tests**: Full integration tests for OpenAI, Google, and Anthropic
- **Structured Output**: JSON response parsing and Pydantic schema validation
- **Caching**: Cache performance and correctness testing
- **Concurrent Requests**: Batch processing and threading
- **Error Handling**: Various edge cases and invalid inputs
- **Retry Logic**: Failure handling and retry mechanisms
- **Parameter Handling**: Config parameters and overrides

## Expected Behavior

- Tests will **skip** providers where API keys are not configured
- Tests will **pass** for basic functionality without API keys
- Cache tests create temporary cache files in `~/.cache/my_llms/`
- Concurrent tests may take longer due to actual API calls

## Notes

- The comprehensive tests make actual API calls and may incur costs
- Tests are designed to be robust and handle missing dependencies gracefully
- All tests clean up after themselves (clear caches, etc.)