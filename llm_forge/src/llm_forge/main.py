#!/usr/bin/env python
"""
Recursive LLM Reasoning Framework with Self-Correction.

This module implements a multi-component system for generating structured
AI responses with recursive refinement. It parses user prompts into structured
requests, generates content for each section, and applies recursive correction
to enhance output quality.

This module provides the complete end-to-end pipeline for the LLM Forge system,
including both the processing logic and a demonstration entry point.
"""

import json
from typing import Final

from llm_forge.logging_config import configure_logging
from llm_forge.nltk_utils import ensure_punkt
from llm_forge.response_loop import process_user_prompt
from llm_forge.type_definitions import ModelResponse

# Ensure required NLTK data is available at import time
ensure_punkt()

# Configure logging with module context
logger: Final = configure_logging()
ensure_punkt()


def main() -> None:
    """
    Execute the main program workflow for the LLM Forge system.

    This function orchestrates the entire process from parsing an example prompt to
    generating structured content, refining it recursively, and outputting
    the final comprehensive response with multiple models and sections.

    The function demonstrates the full pipeline in action using an example
    prompt comparing different LLM architectures.

    Returns:
        None: Results are printed to standard output

    Examples:
        >>> main()
        # Outputs a structured JSON response to stdout comparing LLM architectures
    """
    logger.info("Program started.")

    # Example prompt comparing LLM architectures
    user_prompt: Final[str] = (
        "Compare three different LLM architectures: GPT, Claude, and Mistral. "
        "Provide details on:\n"
        "1. Model architecture\n"
        "2. Training data and methodology\n"
        "3. Strengths & Weaknesses\n"
        "4. Real-world use cases"
    )

    # Process the prompt through the full pipeline
    final_response: ModelResponse = process_user_prompt(user_prompt)

    # Output the fully processed result
    logger.info("Program finished. Final response:")
    print(json.dumps(final_response, indent=2))


if __name__ == "__main__":
    main()
