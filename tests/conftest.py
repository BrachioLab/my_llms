"""Pytest configuration for my_llms tests."""

import sys
import os
from pathlib import Path

# Add src directory to Python path for imports
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Optional: Add fixtures or test configuration here
import pytest

@pytest.fixture(scope="session")
def sample_config():
    """Sample configuration for testing."""
    return {
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "generation_config": {"temperature": 0.7, "max_tokens": 50},
        "use_cache": True
    }