"""Tests for :mod:`llm_forge.input_parser`."""

from llm_forge.input_parser import parse_input
import pytest


def test_parse_basic_prompt_extracts_components():
    """Validate extraction of topic, models, and sections from a simple prompt."""
    prompt = (
        "Compare GPT and Claude:\n"
        "- overview:\n"
        "- technical details:\n"
        "- advantages:\n"
        "- limitations:"
    )
    result = parse_input(prompt)
    assert result["topic"] == "GPT and Claude"
    assert result["models"] == ["gpt", "claude"]
    assert result["sections"] == [
        "overview",
        "technical_details",
        "advantages",
        "limitations",
    ]


def test_defaults_when_models_and_sections_missing():
    """Ensure defaults are provided when none are specified in the prompt."""
    prompt = "Explain the differences between neural networks and decision trees."
    result = parse_input(prompt)
    assert result["models"] == ["gpt", "claude", "llama"]
    assert result["sections"] == [
        "overview",
        "technical_details",
        "advantages",
        "limitations",
    ]


def test_error_on_empty_prompt():
    """Parser should raise :class:`ValueError` for empty input."""
    with pytest.raises(ValueError):
        parse_input("   ")
