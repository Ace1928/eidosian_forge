"""
Core components for Ollama Forge.

This module exposes the primary client and exception classes for interacting with
the Ollama API. It provides a comprehensive interface with robust error handling
and both synchronous and asynchronous operation modes.

Importing this module gives you immediate access to the OllamaClient and all
exception types needed for precise error handling.
"""

from typing import List

# Import client for package-level access
from ollama_forge.src.ollama_forge.core.client import OllamaClient

# Import all exceptions for package-level access
from ollama_forge.src.ollama_forge.core.exceptions import (
    ERROR_CODE_MAP,
    ConnectionError,
    InvalidRequestError,
    ModelNotFoundError,
    OllamaAPIError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaServerError,
    ParseError,
    ServerError,
    StreamingError,
    TimeoutError,
    ValidationError,
    get_exception_for_status,
)

# Define public API
__all__: List[str] = [
    # Client
    "OllamaClient",
    # Primary exceptions
    "OllamaAPIError",
    "ConnectionError",
    "TimeoutError",
    "ModelNotFoundError",
    "ServerError",
    "InvalidRequestError",
    "StreamingError",
    "ParseError",
    "ValidationError",
    # Legacy exceptions (for backward compatibility)
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaServerError",
    # Utilities
    "ERROR_CODE_MAP",
    "get_exception_for_status",
]
