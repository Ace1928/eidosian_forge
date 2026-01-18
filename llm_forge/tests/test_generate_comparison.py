"""Tests for the comparison generation workflow."""

from llm_forge.comparison_generator import generate_comparison
from llm_forge.type_definitions import ModelResponse, StructuredInput


def test_generate_comparison_produces_all_sections() -> None:
    """Ensure every requested model and section exists in the response."""
    structured_input: StructuredInput = {
        "models": ["gpt", "claude"],
        "sections": ["overview", "advantages", "limitations"],
        "raw_prompt": "Compare GPT and Claude models",
        "topic": "Language models",
    }

    result: ModelResponse = generate_comparison(structured_input)

    assert result["topic"] == structured_input["topic"]

    for model in structured_input["models"]:
        assert model in result["models"], f"Missing model '{model}' in response"
        for section in structured_input["sections"]:
            assert (
                section in result["models"][model]
            ), f"Missing section '{section}' for model '{model}'"
            content = result["models"][model][section]
            assert isinstance(content, str) and content
