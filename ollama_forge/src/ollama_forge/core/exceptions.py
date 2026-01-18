#!/usr/bin/env python3
"""
Exception classes for Ollama Forge.

This module defines a precise, hierarchical exception system to provide
detailed and actionable error information with perfect granularity.
Following Eidosian principles of structure as control, contextual integrity,
and precision as style.
"""

from typing import Any, Dict, Optional


class OllamaAPIError(Exception):
    """Base exception for all Ollama API errors."""

    def __init__(self, message: str, response: Optional[Dict[str, Any]] = None):
        self.message = message
        self.response = response
        super().__init__(message)

    def __str__(self) -> str:
        if self.response:
            return f"{self.message} - Response: {self.response}"
        return self.message


class ConnectionError(OllamaAPIError):
    """Raised when a connection to the Ollama server cannot be established."""

    pass


class TimeoutError(OllamaAPIError):
    """Raised when a request to the Ollama server times out."""

    pass


class ModelNotFoundError(OllamaAPIError):
    """Raised when the requested model does not exist."""

    pass


class ServerError(OllamaAPIError):
    """Raised when the Ollama server returns a 5xx error."""

    pass


class InvalidRequestError(OllamaAPIError):
    """Raised when the request is invalid (4xx error)."""

    pass


class StreamingError(OllamaAPIError):
    """Raised when an error occurs during streaming responses."""

    pass


class ParseError(OllamaAPIError):
    """Raised when parsing a response fails."""

    pass


class ValidationError(OllamaAPIError):
    """Raised when input validation fails."""

    pass


class OllamaConnectionError(ConnectionError):
    """Legacy alias for ConnectionError for backwards compatibility."""

    pass


class OllamaModelNotFoundError(ModelNotFoundError):
    """Legacy alias for ModelNotFoundError for backwards compatibility."""

    pass


class OllamaServerError(ServerError):
    """Legacy alias for ServerError for backwards compatibility."""

    pass


# Map HTTP status codes to exception types for clean error handling
ERROR_CODE_MAP: dict[
    int,
    type[InvalidRequestError]
    | type[ModelNotFoundError]
    | type[TimeoutError]
    | type[ServerError],
] = {
    400: InvalidRequestError,
    404: ModelNotFoundError,
    408: TimeoutError,
    429: ServerError,  # Rate limit exceeded
    500: ServerError,
    502: ServerError,
    503: ServerError,
    504: TimeoutError,
}


def get_exception_for_status(
    status_code: int, message: str, response: Optional[Dict[str, Any]] = None
) -> OllamaAPIError:
    """
    Get the appropriate exception class for an HTTP status code.

    Args:
        status_code: HTTP status code
        message: Error message
        response: Optional response data

    Returns:
        Instantiated exception of the appropriate type
    """
    exception_class = ERROR_CODE_MAP.get(status_code, OllamaAPIError)
    return exception_class(message, response)
