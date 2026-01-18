"""
Recursive LLM Reasoning Framework with Self-Correction.

This package provides a multi-component system for generating structured
AI responses with recursive refinement. It parses user prompts into structured
requests, generates content for each section, and applies recursive correction
to enhance output quality.

Components:
    - input_parser: Transforms natural language prompts into structured data
    - content_generator: Creates model-specific content for each section
    - response_loop: Orchestrates the end-to-end generation process
    - logging_config: Provides stylized, contextual logging functionality
    - type_definitions: Contains TypedDict definitions for strict typing

Usage:
    >>> from llm_forge.response_loop import process_user_prompt
    >>> response = process_user_prompt("Compare GPT and Claude models")
    >>> print(response["topic"])
"""

from typing import Final, List

__version__: Final[str] = "0.1.0"

__all__: Final[List[str]] = ["__version__"]
