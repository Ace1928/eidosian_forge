"""Integration tests for word_forge.parser.language_model using real dependencies."""

from __future__ import annotations

import importlib.util

import pytest


# Check if LLM dependencies are available
def _llm_available() -> bool:
    if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
        return False
    try:
        import word_forge.parser.language_model as language_model
    except Exception:
        return False
    return (
        language_model.AutoTokenizer is not None
        and language_model.AutoModelForCausalLM is not None
        and language_model.torch is not None
    )


_LLM_AVAILABLE = _llm_available()

from word_forge.parser.language_model import ModelState

TEST_MODEL = "sshleifer/tiny-gpt2"


@pytest.mark.skipif(not _LLM_AVAILABLE, reason="LLM dependencies (transformers, torch) not installed")
def test_initialize_and_generate_text() -> None:
    state = ModelState(model_name=TEST_MODEL)
    assert state.initialize() is True

    output = state.generate_text("Hello world", max_new_tokens=8)
    assert output is not None
    assert isinstance(output, str)


@pytest.mark.skipif(not _LLM_AVAILABLE, reason="LLM dependencies (transformers, torch) not installed")
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
