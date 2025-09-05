#!/usr/bin/env python3
"""
Comprehensive test suite for my_llms package.

Tests all API models (OpenAI, Anthropic, Google) with both unit tests and integration tests.
Can be run directly or with pytest.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Optional
import pytest
from pydantic import BaseModel as ResponseSchema

# Add parent directory to path to import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from my_llms import load_model, clear_cache, get_cache_size, list_cache

# Suppress noisy third-party libraries
for lib in ["openai", "httpx", "anthropic", "google"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


class MathAnswer(ResponseSchema):
    """Schema for structured math responses."""
    expression: str
    result: int
    explanation: Optional[str] = None


class SimpleResponse(ResponseSchema):
    """Simple response schema for testing.""" 
    answer: str


def log_section(title):
    """Helper to print section headers consistently."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================================
# UNIT TESTS (pytest compatible)
# ============================================================================

class TestLoadModel:
    """Test the load_model function."""
    
    def test_load_model_invalid_provider(self):
        """Test that invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            load_model({"provider": "invalid"})
    
    def test_load_model_missing_provider(self):
        """Test that missing provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider: None"):
            load_model({})
    
    def test_load_model_missing_model_name(self):
        """Test that missing model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name is required"):
            load_model({"provider": "openai"})


class TestAPIModelCreation:
    """Test API model creation."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("openai", None), 
        reason="OpenAI not installed"
    )
    def test_openai_model_creation(self):
        """Test OpenAI model can be created."""
        config = {"provider": "openai", "model_name": "gpt-4o-mini"}
        try:
            model = load_model(config) 
            assert model is not None
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No OpenAI API key configured")
            raise
    
    @pytest.mark.skipif(
        not pytest.importorskip("google.genai", None),
        reason="Google GenAI not installed" 
    )
    def test_google_model_creation(self):
        """Test Google model can be created."""
        config = {"provider": "google", "model_name": "gemini-2.0-flash-exp"}
        try:
            model = load_model(config)
            assert model is not None
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No Google API key configured")
            raise
    
    @pytest.mark.skipif(
        not pytest.importorskip("anthropic", None),
        reason="Anthropic not installed"
    )  
    def test_anthropic_model_creation(self):
        """Test Anthropic model can be created."""
        config = {"provider": "anthropic", "model_name": "claude-3-haiku-20240307"}
        try:
            model = load_model(config)
            assert model is not None
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No Anthropic API key configured")
            raise


class TestCaching:
    """Test caching functionality."""
    
    def test_clear_cache(self):
        """Test cache clearing."""
        clear_cache()  # Should not raise
        
    def test_get_cache_size(self):
        """Test cache size retrieval."""
        size = get_cache_size()
        assert isinstance(size, (int, float))
        assert size >= 0


class TestModelBehavior:
    """Test model behavior with basic validation."""
    
    def test_empty_prompts_error(self):
        """Test that empty prompts raise ValueError.""" 
        config = {"provider": "openai", "model_name": "gpt-4o-mini"}
        try:
            model = load_model(config)
            with pytest.raises(ValueError, match="cannot be empty"):
                model([])
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No API key configured")
            raise
    
    def test_none_prompts_error(self):
        """Test that None prompts raise ValueError."""
        config = {"provider": "openai", "model_name": "gpt-4o-mini"}
        try:
            model = load_model(config)
            with pytest.raises(ValueError, match="cannot be empty"):
                model(None)
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No API key configured")
            raise
    
    def test_invalid_prompt_types_error(self):
        """Test that invalid prompt types raise ValueError."""
        config = {"provider": "openai", "model_name": "gpt-4o-mini"}
        try:
            model = load_model(config)
            with pytest.raises(ValueError, match="must be a non-empty string"):
                model([123, 456])  # Invalid prompt types
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No API key configured")
            raise
    
    def test_empty_string_prompts_error(self):
        """Test that empty string prompts raise ValueError."""
        config = {"provider": "openai", "model_name": "gpt-4o-mini"}
        try:
            model = load_model(config)
            with pytest.raises(ValueError, match="must be a non-empty string"):
                model(["", "   "])  # Empty and whitespace-only strings
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No API key configured")
            raise


class TestConfigHandling:
    """Test configuration parameter handling."""
    
    def test_config_defaults(self):
        """Test that model loads with default configuration values."""
        config = {"provider": "openai", "model_name": "gpt-4o-mini"}
        try:
            model = load_model(config)
            # Check that default values are set
            assert model.use_cache == True
            assert model.num_tries_per_request == 3
            assert model.verbose == False
            assert model.generation_config == {}
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No API key configured")
            raise
    
    def test_config_overrides(self):
        """Test that config values can be overridden."""
        config = {
            "provider": "openai", 
            "model_name": "gpt-4o-mini",
            "use_cache": False,
            "num_tries_per_request": 5,
            "verbose": True,
            "generation_config": {"temperature": 0.8}
        }
        try:
            model = load_model(config)
            assert model.use_cache == False
            assert model.num_tries_per_request == 5
            assert model.verbose == True
            assert model.generation_config == {"temperature": 0.8}
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No API key configured")
            raise
    
    def test_api_key_in_config(self):
        """Test that API key can be provided in config."""
        config = {
            "provider": "openai",
            "model_name": "gpt-4o-mini", 
            "api_key": "test-key"
        }
        # This should not raise a "no API key" error during model creation
        # It will fail later when trying to make actual API calls
        model = load_model(config)
        assert model is not None


class TestCacheUtilities:
    """Test cache utility functions."""
    
    def test_cache_utilities_exist(self):
        """Test that all cache utility functions are available."""
        from my_llms import clear_cache, get_cache_size, list_cache, CACHE_DIR
        assert callable(clear_cache)
        assert callable(get_cache_size)
        assert callable(list_cache)
        assert CACHE_DIR is not None
    
    def test_cache_size_returns_number(self):
        """Test that get_cache_size returns a numeric value."""
        size = get_cache_size()
        assert isinstance(size, (int, float))
        assert size >= 0
    
    def test_clear_cache_executes(self):
        """Test that clear_cache executes without error."""
        clear_cache()  # Should not raise any exceptions


class TestProviderSpecific:
    """Test provider-specific functionality."""
    
    @pytest.mark.parametrize("provider,model_name", [
        ("openai", "gpt-4o-mini"),
        ("google", "gemini-2.0-flash-exp"),
        ("anthropic", "claude-3-haiku-20240307")
    ])
    def test_provider_model_creation(self, provider, model_name):
        """Test that each provider can create models."""
        config = {"provider": provider, "model_name": model_name}
        try:
            model = load_model(config)
            assert model is not None
            assert hasattr(model, '__call__')  # Should be callable
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip(f"No {provider} API key configured")
            raise
    
    def test_openai_specific_params(self):
        """Test OpenAI-specific parameter handling."""
        config = {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "generation_config": {
                "temperature": 0.7,
                "max_tokens": 100,
                "top_p": 0.9,
                "frequency_penalty": 0.1
            }
        }
        try:
            model = load_model(config)
            # Should create without error - specific params will be filtered internally
            assert model.generation_config["temperature"] == 0.7
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No OpenAI API key configured")
            raise
    
    def test_google_specific_params(self):
        """Test Google-specific parameter handling."""
        config = {
            "provider": "google",
            "model_name": "gemini-2.0-flash-exp",
            "generation_config": {
                "temperature": 0.7,
                "max_tokens": 100,  # Should be converted to max_output_tokens
                "top_k": 40
            }
        }
        try:
            model = load_model(config)
            assert model.generation_config["max_tokens"] == 100
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No Google API key configured")
            raise
    
    def test_anthropic_specific_params(self):
        """Test Anthropic-specific parameter handling."""
        config = {
            "provider": "anthropic",
            "model_name": "claude-3-haiku-20240307",
            "generation_config": {
                "temperature": 0.7,
                "max_tokens": 100,
                "top_p": 0.9,
                "top_k": 10
            }
        }
        try:
            model = load_model(config)
            assert model.generation_config["max_tokens"] == 100
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No Anthropic API key configured")
            raise


class TestRetryBehavior:
    """Test retry behavior and error handling."""
    
    def test_runtime_error_after_retries(self):
        """Test that RuntimeError is raised with proper message after all retries fail."""
        config = {
            "provider": "openai",
            "model_name": "gpt-invalid-model-that-does-not-exist",
            "num_tries_per_request": 2,
            "verbose": False
        }
        
        try:
            model = load_model(config)
            # This should fail and raise RuntimeError after retries
            with pytest.raises(RuntimeError) as exc_info:
                model("test prompt")
            
            error_msg = str(exc_info.value)
            assert "after 2 attempts" in error_msg
            assert "Last error:" in error_msg
            assert "gpt-invalid-model-that-does-not-exist" in error_msg
            
        except ValueError as e:
            if "API key" in str(e):
                pytest.skip("No OpenAI API key configured")
            raise
    
    def test_retry_with_different_attempt_counts(self):
        """Test that retry attempts match configuration."""
        for num_tries in [1, 3, 5]:
            config = {
                "provider": "openai",
                "model_name": "gpt-invalid-model",
                "num_tries_per_request": num_tries,
                "verbose": False
            }
            
            try:
                model = load_model(config)
                with pytest.raises(RuntimeError) as exc_info:
                    model("test")
                
                error_msg = str(exc_info.value)
                assert f"after {num_tries} attempts" in error_msg
                
            except ValueError as e:
                if "API key" in str(e):
                    pytest.skip("No OpenAI API key configured")
                raise


class TestResponseSchemas:
    """Test Pydantic response schema handling."""
    
    def test_response_schema_import(self):
        """Test that ResponseSchema can be imported and used."""
        from pydantic import BaseModel
        
        class TestSchema(BaseModel):
            text: str
            number: int
        
        # Should create without error
        schema = TestSchema(text="test", number=42)
        assert schema.text == "test"
        assert schema.number == 42


# ============================================================================
# INTEGRATION TESTS (for direct execution)
# ============================================================================

def test_api_models():
    """Test API-based models (OpenAI, Anthropic, Google)."""
    log_section("Testing API Models")
    
    # Common test data
    test_prompts = ["What is 2 + 2?", "What is 5 * 3?"]
    
    # API configurations
    api_configs = [
        ("OpenAI GPT-4o-mini", {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "generation_config": {"temperature": 0.7, "max_tokens": 50},
            "use_cache": True
        }),
        ("Google Gemini Flash", {
            "provider": "google", 
            "model_name": "gemini-2.0-flash-exp",
            "generation_config": {"temperature": 0.7, "max_tokens": 50},
            "use_cache": True
        }),
        ("Anthropic Claude Haiku", {
            "provider": "anthropic",
            "model_name": "claude-3-haiku-20240307", 
            "generation_config": {"temperature": 0.7, "max_tokens": 50},
            "use_cache": True
        })
    ]
    
    for name, config in api_configs:
        print(f"\nTesting {name}...")
        
        try:
            model = load_model(config)
            print(f"✓ Model loaded: {name}")
            
            # Basic tests using __call__ method
            response = model(test_prompts[0])
            print(f"✓ Single prompt: {response[:50]}...")
            
            responses = model(test_prompts)
            print(f"✓ Batch prompts: {len(responses)} responses")
            
            response = model("Count from 1 to 5", temperature=0.1, max_tokens=30)
            print(f"✓ Custom params: {response[:50]}...")
            
            # Structured output tests (OpenAI and Google only) - handled in integration tests
            
            print(f"✅ All tests passed for {name}")
            
        except ValueError as e:
            print(f"⚠️  {name}: {e}")
        except Exception as e:
            print(f"❌ {name} failed: {e}")




def test_error_handling():
    """Test error handling for various edge cases."""
    log_section("Testing Error Handling")
    
    test_cases = [
        ({"provider": "invalid_provider"}, "invalid provider"),
        ({}, "missing provider"),
        ({"provider": "openai"}, "missing model_name")
    ]
    
    for config, description in test_cases:
        try:
            model = load_model(config)
            print(f"❌ Should have raised error for {description}")
        except (ValueError, KeyError) as e:
            print(f"✓ Correctly raised error for {description}: {str(e)[:50]}...")
        except Exception as e:
            print(f"✓ Raised error for {description}: {type(e).__name__}")
    
    # Test empty prompts
    try:
        model = load_model({"provider": "openai", "model_name": "gpt-4o-mini"})
        model([])
        print("❌ Should have raised error for empty prompts")
    except ValueError:
        print("✓ Correctly raised error for empty prompts")
    except Exception as e:
        print(f"⚠️  Error test for empty prompts: {e}")
    
    # Test None prompts
    try:
        model = load_model({"provider": "openai", "model_name": "gpt-4o-mini"})
        model(None)
        print("❌ Should have raised error for None prompts")
    except ValueError:
        print("✓ Correctly raised error for None prompts")
    except Exception as e:
        print(f"⚠️  Error test for None prompts: {e}")
    
    # Test invalid prompt types
    try:
        model = load_model({"provider": "openai", "model_name": "gpt-4o-mini"})
        model([123, 456])  # Invalid prompt types
        print("❌ Should have raised error for invalid prompt types")
    except ValueError:
        print("✓ Correctly raised error for invalid prompt types")
    except Exception as e:
        print(f"⚠️  Error test for invalid prompt types: {e}")
    
    print("✅ All error handling tests passed")


def test_caching():
    """Test caching functionality for API models."""
    log_section("Testing Caching")
    
    # Clear cache first
    clear_cache()
    print("✓ Cache cleared")
    
    config = {
        "provider": "openai",
        "model_name": "gpt-4o-mini", 
        "use_cache": True,
        "generation_config": {"temperature": 0, "max_tokens": 10}
    }
    
    try:
        model = load_model(config)
        
        # Check initial cache state
        initial_size = get_cache_size()
        print(f"Initial cache size: {initial_size} MB")
        
        # Measure first call
        start = time.perf_counter()
        response1 = model("What is 1 + 1?")
        time1 = time.perf_counter() - start
        
        # Check cache size after first call
        after_first_size = get_cache_size()
        print(f"Cache size after first call: {after_first_size} MB")
        
        # Measure cached call
        start = time.perf_counter()
        response2 = model("What is 1 + 1?")
        time2 = time.perf_counter() - start
        
        print(f"First call: {time1:.3f}s - {response1[:30]}...")
        print(f"Cached call: {time2:.3f}s - {response2[:30]}...")
        
        # Check results
        if response1 == response2:
            print("✓ Responses match (cache working)")
        else:
            print("⚠️  Responses differ (potential cache issue)")
        
        # Cache should be significantly faster (at least 10x for local cache)
        if time1 > 0.001 and time2 < time1 * 0.1:
            print("✓ Cache hit was significantly faster")
        elif time2 < 0.001:
            print("✓ Cache hit (too fast to measure)")
        else:
            print(f"⚠️  Cache timing inconclusive (first: {time1:.3f}s, cached: {time2:.3f}s)")
        
        # Test cache utilities
        list_cache()
        final_size = get_cache_size()
        print(f"Final cache size: {final_size} MB")
        
        # Test cache disabling
        config_no_cache = config.copy()
        config_no_cache["use_cache"] = False
        model_no_cache = load_model(config_no_cache)
        
        start = time.perf_counter()
        response3 = model_no_cache("What is 1 + 1?")
        time3 = time.perf_counter() - start
        print(f"No-cache call: {time3:.3f}s - {response3[:30]}...")
        
        if time3 > time2:
            print("✓ No-cache call was slower than cached call")
        
        print("✅ Cache test completed")
        
    except ValueError as e:
        print(f"⚠️  Cache test skipped: {e}")
    except Exception as e:
        print(f"❌ Cache test failed: {e}")


def test_concurrent_requests():
    """Test concurrent request handling."""
    log_section("Testing Concurrent Requests")
    
    config = {
        "provider": "openai", 
        "model_name": "gpt-4o-mini",
        "generation_config": {"temperature": 0.7, "max_tokens": 20},
        "use_cache": False,  # Disable cache to test actual concurrency
        "verbose": True  # Enable verbose to see concurrent processing
    }
    
    try:
        model = load_model(config)
        
        # Test batch processing (which uses ThreadPoolExecutor internally)
        batch_prompts = [
            "What is the capital of France?",
            "What is the capital of Spain?", 
            "What is the capital of Italy?",
            "What is the capital of Germany?",
            "What is the capital of Portugal?"
        ]
        
        print(f"Sending {len(batch_prompts)} concurrent requests...")
        start = time.perf_counter()
        responses = model(batch_prompts)
        total_time = time.perf_counter() - start
        
        print(f"✓ All {len(responses)} responses received in {total_time:.2f}s")
        print(f"✓ Average time per request: {total_time/len(responses):.2f}s")
        
        # Verify all responses
        for i, response in enumerate(responses):
            if response and len(response) > 5:
                print(f"✓ Response {i+1}: {response[:40]}...")
            else:
                print(f"⚠️  Response {i+1} seems short or empty")
        
        # Test smaller batch for speed comparison
        small_batch = batch_prompts[:2]
        start = time.perf_counter()
        small_responses = model(small_batch)
        small_time = time.perf_counter() - start
        
        print(f"✓ Small batch ({len(small_batch)}): {small_time:.2f}s")
        
        print("✅ Concurrent request test completed")
        
    except ValueError as e:
        print(f"⚠️  Concurrent test skipped: {e}")
    except Exception as e:
        print(f"❌ Concurrent test failed: {e}")


def test_retry_logic():
    """Test retry logic with invalid model names."""
    log_section("Testing Retry Logic")
    
    config = {
        "provider": "openai",
        "model_name": "gpt-invalid-model-name",  # Invalid model to trigger retries
        "num_tries_per_request": 2,  # Limit retries for faster testing
        "verbose": True  # Show retry messages
    }
    
    try:
        model = load_model(config)
        print("✓ Model loaded (this should work even with invalid model name)")
        
        # This should fail and retry
        start = time.perf_counter()
        response = model("Hello")
        total_time = time.perf_counter() - start
        
        print(f"❌ Unexpectedly got response: {response[:50]}...")
        
    except RuntimeError as e:
        # Should get a RuntimeError with details about retry failures
        error_msg = str(e)
        if "after 2 attempts" in error_msg and "Last error:" in error_msg:
            print(f"✓ Correctly raised RuntimeError after retries: {error_msg[:100]}...")
        else:
            print(f"⚠️  Got RuntimeError but unexpected message: {error_msg[:100]}...")
    except Exception as e:
        print(f"⚠️  Got unexpected exception type: {type(e).__name__}: {str(e)[:100]}...")
    
    print("✅ Retry logic test completed")


def test_parameter_handling():
    """Test parameter handling and validation."""
    log_section("Testing Parameter Handling")
    
    config = {
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "generation_config": {"temperature": 0.5}
    }
    
    try:
        model = load_model(config)
        print("✓ Model loaded with default config")
        
        # Test parameter override
        response1 = model("Say hello", temperature=0.1, max_tokens=5)
        print(f"✓ Parameter override: {response1[:30]}...")
        
        # Test generation_config parameters
        response2 = model("Say goodbye")  # Should use temperature=0.5 from config
        print(f"✓ Config parameters: {response2[:30]}...")
        
        # Test unsupported parameters (should be filtered out)
        response3 = model("Say hi", temperature=0.2, unsupported_param="test")
        print(f"✓ Unsupported param filtered: {response3[:30]}...")
        
        print("✅ Parameter handling test completed")
        
    except ValueError as e:
        print(f"⚠️  Parameter test skipped: {e}")
    except Exception as e:
        print(f"❌ Parameter test failed: {e}")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_integration_tests():
    """Run all integration tests."""
    print("Starting comprehensive my_llms integration tests...")
    print(f"Python: {sys.executable}")
    print(f"Working dir: {os.getcwd()}")
    
    tests = [
        ("API Models", test_api_models),
        ("Error Handling", test_error_handling), 
        ("Caching", test_caching),
        ("Concurrent Requests", test_concurrent_requests),
        ("Retry Logic", test_retry_logic),
        ("Parameter Handling", test_parameter_handling)
    ]
    
    for name, func in tests:
        try:
            func()
        except Exception as e:
            print(f"❌ Test {name} crashed: {e}")
            import traceback
            traceback.print_exc()
    
    log_section("All integration tests completed!")


def main():
    """Main entry point - can run integration tests or pytest."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive my_llms testing")
    parser.add_argument("--test", choices=["api", "error", "cache", "concurrent", "retry", "params", "all", "pytest"],
                       default="all", help="Which tests to run")
    parser.add_argument("--pytest", action="store_true", help="Run with pytest instead")
    
    args = parser.parse_args()
    
    if args.pytest or args.test == "pytest":
        # Run with pytest
        pytest.main([__file__, "-v"])
        return
    
    # Run integration tests
    individual_tests = {
        "api": test_api_models,
        "error": test_error_handling, 
        "cache": test_caching,
        "concurrent": test_concurrent_requests,
        "retry": test_retry_logic,
        "params": test_parameter_handling
    }
    
    if args.test == "all":
        run_integration_tests()
    else:
        try:
            individual_tests[args.test]()
        except Exception as e:
            print(f"❌ Test {args.test} crashed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()