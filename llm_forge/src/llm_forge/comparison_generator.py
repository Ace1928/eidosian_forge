"""
Comparison generator for structured analysis of language models.

This module transforms structured input into comprehensive model comparisons
by orchestrating content generation, organizing the results, and ensuring
complete coverage of requested sections across all models.
"""

import re
from typing import Dict, Final, List, Optional

from llm_forge.content_generator import ensure_complete_response, generate_response
from llm_forge.logging_config import configure_logging
from llm_forge.type_definitions import ModelResponse, StructuredInput

# Configure module-specific logger
logger: Final = configure_logging()


def generate_comparison(structured_input: StructuredInput) -> ModelResponse:
    """
    Generate a structured comparison between language models.

    Creates a comprehensive comparison of the specified models across
    requested sections, producing consistent content that highlights
    each model's unique approach to the given topic.

    Args:
        structured_input: Parsed and validated user prompt with models,
                         sections, and topic information

    Returns:
        ModelResponse: Fully populated content for each model and section

    Raises:
        ValueError: If the structured input is invalid or incomplete

    Examples:
        >>> input_data = {"models": ["gpt-4", "claude-2"], "sections": ["overview"], "topic": "AI safety"}
        >>> response = generate_comparison(input_data)
        >>> print(response["models"]["gpt-4"]["overview"])
    """
    # Validate input structure
    _validate_comparison_input(structured_input)

    # Log comparison generation start with context
    logger.info(
        f"Generating comparison for {len(structured_input['models'])} models on topic: {structured_input['topic']}"
    )
    logger.debug(f"Models: {', '.join(structured_input['models'])}")
    logger.debug(f"Sections: {', '.join(structured_input['sections'])}")

    # Generate initial response with content for all models and sections
    initial_response: ModelResponse = generate_response(structured_input)

    # Ensure all required sections are populated
    complete_response: ModelResponse = ensure_complete_response(
        initial_response, structured_input
    )

    # Log completion of comparison generation
    model_count: int = len(complete_response["models"])
    section_count: int = sum(
        len(sections) for sections in complete_response["models"].values()
    )
    logger.info(
        f"Comparison generated: {model_count} models, {section_count} total sections"
    )

    return complete_response


def _validate_comparison_input(structured_input: StructuredInput) -> None:
    """
    Validate structured input for comparison generation.

    Ensures the input contains required fields with valid values before
    proceeding with comparison generation.

    Args:
        structured_input: The parsed input to validate

    Raises:
        ValueError: If input is missing required fields or contains invalid values
    """
    # Check for required fields
    if not structured_input.get("models"):
        raise ValueError("Comparison requires at least one model to analyze")

    if not structured_input.get("sections"):
        raise ValueError("Comparison requires at least one section to generate")

    if not structured_input.get("topic"):
        raise ValueError("Comparison requires a topic to analyze")

    # Validate model names (prevent injection or invalid names)
    # Model names should only contain alphanumeric chars, underscores, hyphens, and periods
    for model in structured_input["models"]:
        if not model or not re.match(r"^[a-zA-Z0-9_\-\.]+$", model):
            raise ValueError(f"Invalid model name: {model}")

    # Enforce reasonable limits
    if len(structured_input["models"]) > 10:
        raise ValueError(
            f"Too many models requested: {len(structured_input['models'])}. Maximum is 10."
        )

    if len(structured_input["sections"]) > 15:
        raise ValueError(
            f"Too many sections requested: {len(structured_input['sections'])}. Maximum is 15."
        )


def extract_differences(
    comparison: ModelResponse, section: str, models: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Extract key differences between models for a specific section.

    Analyzes the content across models for the given section and
    identifies distinctive approaches, capabilities, or limitations.

    Args:
        comparison: The complete model comparison data
        section: The specific section to analyze for differences
        models: Optional list of models to restrict the analysis to

    Returns:
        Dict[str, List[str]]: Dictionary mapping models to lists of distinctive characteristics

    Note:
        This is a simplified implementation. A full implementation would use
        NLP techniques for more sophisticated difference extraction.

    Examples:
        >>> differences = extract_differences(comparison_data, "advantages", ["gpt-4", "claude-2"])
        >>> for model, points in differences.items():
        ...     print(f"{model}: {len(points)} distinctive points")
    """
    differences: Dict[str, List[str]] = {}
    target_models: List[str] = models or list(comparison["models"].keys())

    # Simple extraction based on key phrases and unique content
    # In a real implementation, this would use NLP for semantic analysis
    for model in target_models:
        if model not in comparison["models"]:
            continue

        model_sections: Dict[str, str] = comparison["models"][model]
        if section not in model_sections:
            continue

        content: str = model_sections[section]

        # Simple extraction of sentences with distinctive markers
        # A real implementation would use more sophisticated algorithms
        distinctive_markers: List[str] = [
            f"{model} is",
            "uniquely",
            "specifically",
            "unlike other models",
            "stands out",
            "excels at",
        ]

        # Extract sentences containing distinctive markers
        differences[model] = []
        sentences: List[str] = content.split(". ")
        for sentence in sentences:
            if any(marker in sentence.lower() for marker in distinctive_markers):
                differences[model].append(sentence)

    return differences
