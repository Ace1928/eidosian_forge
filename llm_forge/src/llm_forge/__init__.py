"""
LLM Forge - Unified LLM Interface.

Provides abstraction over multiple LLM providers with caching,
comparison generation, and input parsing capabilities.
"""

from .core.manager import ModelManager
from .core.interfaces import LLMProvider, LLMResponse, EmbeddingProvider
from .providers.openai_provider import OpenAIProvider
from .providers.ollama_provider import OllamaProvider
from .caching.sqlite_cache import SQLiteCache
from .comparison_generator import generate_comparison
from .input_parser import parse_input, validate_input
from .type_definitions import StructuredInput, ModelResponse, ParsedInput

__all__ = [
    "ModelManager", 
    "LLMProvider", 
    "LLMResponse", 
    "EmbeddingProvider", 
    "OpenAIProvider", 
    "OllamaProvider", 
    "SQLiteCache",
    "generate_comparison",
    "parse_input",
    "validate_input",
    "StructuredInput",
    "ModelResponse",
    "ParsedInput",
]