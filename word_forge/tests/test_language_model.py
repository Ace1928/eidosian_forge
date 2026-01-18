"""Integration tests for word_forge.parser.language_model using real dependencies."""

from __future__ import annotations

import importlib.util

import pytest

# Check if LLM dependencies are available
_LLM_AVAILABLE = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("torch") is not None
)

from word_forge.parser.language_model import ModelState

TEST_MODEL = "sshleifer/tiny-gpt2"


@pytest.mark.skipif(
    not _LLM_AVAILABLE, reason="LLM dependencies (transformers, torch) not installed"
)
def test_initialize_and_generate_text() -> None:
    state = ModelState(model_name=TEST_MODEL)
    assert state.initialize() is True

    output = state.generate_text("Hello world", max_new_tokens=8)
    assert output is not None
    assert isinstance(output, str)


@pytest.mark.skipif(
    not _LLM_AVAILABLE, reason="LLM dependencies (transformers, torch) not installed"
)
def test_query_uses_model() -> None:
    state = ModelState(model_name=TEST_MODEL)
    assert state.initialize() is True

    output = state.query("Tell me a short phrase.", max_new_tokens=8)
    assert output is not None
    assert isinstance(output, str)


def test_set_model_resets_initialization() -> None:
    state = ModelState(model_name=TEST_MODEL)
    state._initialized = True
    state.set_model(TEST_MODEL)
    assert state._initialized is False


def test_invalid_model_triggers_failure_tracking() -> None:
    state = ModelState(model_name="invalid/does-not-exist")
    result = state.initialize()
    assert result is False
    assert state._inference_failures >= 1
