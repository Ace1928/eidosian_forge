"""Integration and unit tests for word_forge.parser.language_model."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

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

from word_forge.parser.language_model import ModelState, _clean_llama_cli_output

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


def test_clean_llama_cli_output_extracts_response() -> None:
    raw_output = """
Loading model...

build      : test
model      : demo.gguf
modalities : text

> Prompt line one
Prompt line two

```json
{"phrases": ["atlas"]}
```

Exiting...
"""
    assert _clean_llama_cli_output(raw_output) == '{"phrases": ["atlas"]}'


def test_initialize_gguf_backend(tmp_path: Path) -> None:
    model_path = tmp_path / "mini.gguf"
    model_path.write_text("stub")
    cli_path = tmp_path / "llama-cli"
    cli_path.write_text("#!/bin/sh\n")
    cli_path.chmod(0o755)

    with patch("word_forge.parser.language_model._resolve_llama_cli_path", return_value=cli_path):
        state = ModelState(model_name=f"gguf:{model_path}")
        assert state.initialize() is True
        assert state._gguf_model_path == model_path
        assert state._llama_cli_path == cli_path


def test_generate_text_gguf_backend_uses_cleaned_output(tmp_path: Path) -> None:
    model_path = tmp_path / "mini.gguf"
    model_path.write_text("stub")
    cli_path = tmp_path / "llama-cli"
    cli_path.write_text("#!/bin/sh\n")
    cli_path.chmod(0o755)

    with patch("word_forge.parser.language_model._resolve_llama_cli_path", return_value=cli_path):
        state = ModelState(model_name=f"gguf:{model_path}")
        assert state.initialize() is True

    completed = subprocess.CompletedProcess(
        args=[str(cli_path)],
        returncode=0,
        stdout='''Loading model...\n\n> Return JSON\n\n{"ipa":"x","arpabet":"X","stress_pattern":"1"}\n\nExiting...\n''',
        stderr="",
    )
    with patch("word_forge.parser.language_model.subprocess.run", return_value=completed) as mocked_run:
        output = state.generate_text("Return JSON", max_new_tokens=32)
    assert output == '{"ipa":"x","arpabet":"X","stress_pattern":"1"}'
    mocked_run.assert_called_once()
