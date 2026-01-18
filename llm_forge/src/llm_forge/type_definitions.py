"""
Type definitions for the LLM Forge system.

This module contains TypedDict, Protocol, and other type definitions used throughout
the LLM Forge system to ensure type safety and consistency.
"""

from typing import Dict, List, Literal, Protocol, TypedDict, runtime_checkable

# Standard section types for content generation
SectionType = Literal[
    "overview",
    "technical_details",
    "advantages",
    "limitations",
    "model_architecture",
    "training_data",
    "strengths_weaknesses",
    "use_cases",
]

# Model types supported by the system
ModelType = Literal["gpt", "claude", "llama", "mistral", "gemini", "palm", "falcon"]


class StructuredInput(TypedDict):
    """
    Structured representation of a parsed user prompt.

    Attributes:
        models: List of model names to generate responses for
        sections: List of section names to include in responses
        raw_prompt: The original unmodified user prompt
        topic: The main subject extracted from the prompt
    """

    models: List[str]
    sections: List[str]
    raw_prompt: str
    topic: str


class ModelContent(TypedDict):
    """
    Complete content for a specific model.

    A dictionary mapping section names to their content.
    """

    pass


class ModelResponse(TypedDict):
    """
    Structured representation of generated model responses.

    Attributes:
        topic: The main subject of the response
        models: Dictionary mapping model names to their section responses
    """

    topic: str
    models: Dict[str, Dict[str, str]]


@runtime_checkable
class ContentGenerator(Protocol):
    """
    Protocol defining the interface for content generation components.

    Any class implementing this protocol can be used as a content generator
    in the LLM Forge system.
    """

    def generate(self, model: str, section: str, topic: str) -> str:
        """
        Generate content for a specific model and section.

        Args:
            model: The model to generate content for
            section: The section to generate
            topic: The topic to generate content about

        Returns:
            Generated content as a string
        """
        ...
