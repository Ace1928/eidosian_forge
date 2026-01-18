"""
Parser for structuring user input into a machine-readable format.

This module contains functions to analyze and transform natural language
user prompts into structured representations that can be processed by the
LLM system for content generation.
"""

import re
from typing import Final, List, Match, Optional, cast

# Type stub for nltk.tokenize functions
from nltk import download
from nltk.tokenize import sent_tokenize  # type: ignore

from llm_forge.logging_config import configure_logging
from llm_forge.type_definitions import StructuredInput

# Ensure punkt tokenizer is available
download("punkt", quiet=True)
download("punkt_tab", quiet=True)

logger: Final = configure_logging()


def parse_input(user_prompt: str) -> StructuredInput:
    """
    Parse user input into a structured format for processing.

    Analyzes the natural language prompt to extract:
    - The main topic
    - Referenced models (defaults to ['gpt', 'claude', 'llama'] if none detected)
    - Content sections (defaults to standard sections if none specified)

    Args:
        user_prompt: The raw text prompt from the user

    Returns:
        A StructuredInput object containing the extracted information

    Raises:
        ValueError: If the prompt is empty or cannot be parsed
    """
    if not user_prompt.strip():
        logger.error("Empty prompt received")
        raise ValueError("The prompt cannot be empty")

    # Extract topic from the prompt
    topic: str = _extract_topic(user_prompt)
    logger.debug(f"Extracted topic: {topic}")

    # Extract model names, defaulting to standard models if none found
    models: List[str] = _extract_models(user_prompt) or ["gpt", "claude", "llama"]
    logger.debug(f"Identified models: {models}")

    # Extract content sections, defaulting to standard sections if none found
    sections: List[str] = _extract_sections(user_prompt) or [
        "overview",
        "technical_details",
        "advantages",
        "limitations",
    ]
    logger.debug(f"Identified sections: {sections}")

    # Create and return the structured input
    return StructuredInput(
        models=models, sections=sections, raw_prompt=user_prompt, topic=topic
    )


def _extract_topic(prompt: str) -> str:
    """
    Extract the main topic from the user prompt.

    Uses sentence tokenization to identify the first sentence and
    extracts key noun phrases as the topic.

    Args:
        prompt: The user prompt text

    Returns:
        The extracted topic as a string
    """
    # Simple implementation - in a real system, this would use more
    # sophisticated NLP techniques to extract the actual topic
    sentences: List[str] = sent_tokenize(prompt) if prompt else []
    first_sentence: str = sentences[0] if sentences else prompt

    # Look for comparison patterns
    compare_match: Optional[Match[str]] = re.search(
        r"compare\s+(\w+(?:\s+\w+)*)", first_sentence, re.IGNORECASE
    )
    if compare_match and compare_match.group(1):
        return cast(str, compare_match.group(1))

    # Default extraction from first sentence
    words: List[str] = first_sentence.split()
    if len(words) > 5:
        return " ".join(words[:5]) + "..."
    return first_sentence


def _extract_models(prompt: str) -> List[str]:
    """
    Extract model names mentioned in the prompt.

    Searches for common LLM model names in the prompt text.

    Args:
        prompt: The user prompt text

    Returns:
        List of model names found in the prompt, or empty list if none found
    """
    # Common model keywords to look for
    model_keywords: Final[List[str]] = [
        "gpt",
        "llama",
        "claude",
        "mistral",
        "gemini",
        "palm",
        "bard",
        "bert",
        "t5",
        "falcon",
    ]

    found_models: List[str] = []
    lower_prompt: str = prompt.lower()

    for model in model_keywords:
        if model in lower_prompt:
            found_models.append(model)

    return found_models


def _extract_sections(prompt: str) -> List[str]:
    """
    Extract content sections requested in the prompt.

    Looks for numbered lists, bullet points, or keywords indicating
    specific sections to include in the response.

    Args:
        prompt: The user prompt text

    Returns:
        List of section names identified in the prompt, or empty list if none found
    """
    # Check for numbered or bulleted lists
    list_pattern: Final[str] = r"\n\s*(?:[0-9]+\.|\-|\*)\s*([^:]+)(?::)?"
    list_matches: List[str] = re.findall(list_pattern, prompt)

    if list_matches:
        # Convert the matches to standardized section names
        return [_standardize_section_name(match.strip()) for match in list_matches]

    return []


def _standardize_section_name(section: str) -> str:
    """
    Convert a natural language section name to a standardized format.

    Args:
        section: The raw section name from the prompt

    Returns:
        A standardized, snake_case section identifier
    """
    # Remove any non-alphanumeric characters and convert to lowercase
    cleaned: str = re.sub(r"[^a-zA-Z0-9\s]", "", section.lower())
    # Replace spaces with underscores
    return re.sub(r"\s+", "_", cleaned.strip())
