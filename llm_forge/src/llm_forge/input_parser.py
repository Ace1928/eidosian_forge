from eidosian_core import eidosian
"""
Input Parser for LLM Forge.

Parses natural language or structured input into actionable requests.
"""

import re
from typing import Dict, Any, List, Optional
from .type_definitions import ParsedInput


# Default sections for comparison
DEFAULT_SECTIONS = ["overview", "technical_details", "advantages", "limitations"]
DEFAULT_MODELS = ["gpt", "claude", "llama"]


@eidosian()
def parse_input(raw_input: str) -> ParsedInput:
    """
    Parse raw input string into structured format.
    
    Args:
        raw_input: User's raw input text
        
    Returns:
        ParsedInput with prompt, models, topic, sections, and options extracted
        
    Raises:
        ValueError: If input is empty or whitespace-only
    """
    stripped = raw_input.strip()
    if not stripped:
        raise ValueError("Input cannot be empty")
    
    result: ParsedInput = {
        "prompt": stripped,
        "models": [],
        "sections": [],
        "options": {},
        "errors": [],
    }
    
    # Extract topic from "Compare X and Y:" pattern
    topic_match = re.search(r'[Cc]ompare\s+(.+?):', stripped)
    if topic_match:
        result["topic"] = topic_match.group(1).strip()
    
    # Look for model mentions
    input_lower = stripped.lower()
    known_models = ["gpt", "claude", "llama", "phi", "qwen", "gemini", "mistral"]
    
    for model in known_models:
        if model in input_lower:
            result["models"].append(model)
    
    # If no models found, use defaults
    if not result["models"]:
        result["models"] = DEFAULT_MODELS.copy()
    
    # Extract sections from "- section:" pattern
    section_matches = re.findall(r'-\s*(\w+(?:\s+\w+)*):', stripped)
    if section_matches:
        # Convert to snake_case
        for section in section_matches:
            snake_section = section.lower().replace(' ', '_')
            result["sections"].append(snake_section)
    else:
        # Use defaults
        result["sections"] = DEFAULT_SECTIONS.copy()
    
    return result


@eidosian()
def validate_input(parsed: ParsedInput) -> bool:
    """
    Validate parsed input for completeness.
    
    Returns:
        True if input is valid, False otherwise
    """
    if not parsed.get("prompt"):
        return False
    if not parsed.get("models"):
        return False
    return True


__all__ = ["parse_input", "validate_input"]
