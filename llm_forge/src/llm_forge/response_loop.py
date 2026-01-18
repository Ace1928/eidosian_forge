#!/usr/bin/env python
"""
Recursive LLM Reasoning Framework with Self-Correction.

This module implements a multi-component system for generating structured
AI responses with recursive refinement. It parses user prompts into structured
requests, generates content for each section, and applies recursive correction
to enhance output quality.
"""

import json
from typing import Final

from llm_forge.content_generator import ensure_complete_response, generate_response
from llm_forge.input_parser import parse_input
from llm_forge.logging_config import configure_logging
from llm_forge.nltk_utils import ensure_punkt
from llm_forge.type_definitions import ModelResponse, StructuredInput

# Ensure required NLTK data is available at import time
ensure_punkt()

# Configure logging with module context
logger: Final = configure_logging()


def process_user_prompt(user_prompt: str) -> ModelResponse:
    """
    Process a user prompt into a structured response.

    This is the main orchestration function that coordinates the entire process
    from parsing the prompt to generating and finalizing the response. The function
    follows a three-step pipeline:

    1. Parse the prompt into structured data
    2. Generate initial content for each model and section
    3. Validate and complete any missing sections

    Args:
        user_prompt: The raw user input text to process

    Returns:
        A structured response with content for each model and section

    Raises:
        ValueError: If the prompt cannot be properly parsed or is empty

    Examples:
        >>> response = process_user_prompt("Compare GPT and Claude models")
        >>> print(response["topic"])
        "GPT and Claude models"
    """
    # Step 1: Parse the prompt into structured components
    logger.info("Parsing user prompt...")
    structured_input: StructuredInput = parse_input(user_prompt)
    logger.debug("Structured Input:\n" + json.dumps(structured_input, indent=2))

    # Step 2: Generate an initial response (with refinement & template formatting)
    logger.info("Generating initial AI response...")
    initial_response: ModelResponse = generate_response(structured_input)
    logger.debug("Initial Response:\n" + json.dumps(initial_response, indent=2))

    # Step 3: Recursively check and fill in any missing sections
    logger.info("Validating and completing AI response...")
    final_response: ModelResponse = ensure_complete_response(
        initial_response, structured_input
    )
    logger.debug("Final Completed Response:\n" + json.dumps(final_response, indent=2))

    return final_response
