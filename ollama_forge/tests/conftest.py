#!/usr/bin/env python3
"""
Test configuration and fixtures for pytest.

This file contains common fixtures and configuration settings
for the test suite.
"""
import pytest

# Sample test data
@pytest.fixture
def sample_chat_messages():
    """Return a list of chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def sample_embeddings():
    """Return sample embeddings for testing."""
    return [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]


@pytest.fixture
def sample_api_response():
    """Return a sample API response for testing."""
    return {
        "model": "test-model",
        "created_at": "2023-08-04T19:22:45.499127Z",
        "response": "This is a test response.",
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 1000000000,
        "prompt_eval_count": 10,
        "eval_count": 20,
    }

@pytest.fixture(scope="session")
def ollama_client():
    from ollama_forge import OllamaClient
    return OllamaClient()
