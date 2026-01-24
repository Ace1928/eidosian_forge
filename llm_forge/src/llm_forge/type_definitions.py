"""
Type definitions for LLM Forge.

Provides type aliases and TypedDicts for the llm_forge ecosystem.
"""

from typing import Dict, Any, List, TypedDict, Optional


class StructuredInput(TypedDict, total=False):
    """Input structure for comparison generator."""
    models: List[str]
    sections: List[str]
    raw_prompt: str
    topic: str


class ModelSectionContent(TypedDict):
    """Content for a specific model/section combination."""
    content: str
    tokens_used: int
    latency_ms: float


class ModelResponse(TypedDict):
    """Response structure from comparison generator."""
    topic: str
    models: Dict[str, Dict[str, str]]


class ParsedInput(TypedDict, total=False):
    """Result from input parser."""
    prompt: str
    models: List[str]
    sections: List[str]
    topic: str
    options: Dict[str, Any]
    errors: List[str]


__all__ = [
    "StructuredInput",
    "ModelSectionContent", 
    "ModelResponse",
    "ParsedInput",
]
