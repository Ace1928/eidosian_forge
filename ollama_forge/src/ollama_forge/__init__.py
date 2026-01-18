# 🌀 Eidosian seed of the package – minimal, universal, ever-evolving! 🌱

"""
Ollama Forge - Python client library and CLI for Ollama

This package provides a comprehensive set of helpers for interacting with Ollama,
following all ten Eidosian principles for elegant, efficient, and effective code.

Key Features:
- Complete API coverage
- Synchronous and asynchronous interfaces
- Smart model fallbacks
- CLI helpers
- Embedding utilities
"""

import os
import warnings
from typing import Any, Dict, Type, TypeVar, cast

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📦 Version and Configuration - Adaptive Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Use variables for constants to avoid redefinition issues
_version = "0.1.9"  # Fallback version
_author = "Lloyd Handyside, Eidos"
_email = "ace1928@gmail.com, syntheticeidos@gmail.com"
_ollama_api_url = "http://localhost:11434"
_chat_model = "deepseek-r1:1.5b"  # Best small model for chat
_backup_chat_model = "qwen2.5:0.5b-Instruct"  # Excellent fallback
_embedding_model = _chat_model  # Using chat model for embeddings improves context
_backup_embedding_model = _backup_chat_model  # Same fallback for embeddings


from .config import (
    BACKUP_CHAT_MODEL,
    BACKUP_EMBEDDING_MODEL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_API_URL,
    get_author_string,
    get_email_string,
    get_version_string,
)
from .config import VERSION as __version__

__author__ = get_author_string()
__email__ = get_email_string()

# Update our variables with imported values
_version = __version__
_author = __author__
_email = __email__
_ollama_api_url = DEFAULT_OLLAMA_API_URL
_chat_model = DEFAULT_CHAT_MODEL
_backup_chat_model = BACKUP_CHAT_MODEL
_embedding_model = DEFAULT_EMBEDDING_MODEL
_backup_embedding_model = BACKUP_EMBEDDING_MODEL
# License and URL information
__license__ = "MIT"
__url__ = "https://github.com/Ace1928/ollama_forge"
__description__ = "Python client library and CLI for Ollama"


# Functions to access constants instead of exposing them directly
# This prevents constant redefinition while keeping the API consistent
def get_default_api_url() -> str:
    return _ollama_api_url


def get_default_chat_model() -> str:
    return _chat_model


def get_backup_chat_model() -> str:
    return _backup_chat_model


def get_default_embedding_model() -> str:
    return _embedding_model


def get_backup_embedding_model() -> str:
    return _backup_embedding_model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧰 Core Components - Elegant Import Strategy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Type definition for client classes
T = TypeVar("T")
ClientType = Type[T]  # Represents a client class type


# Define placeholders for components that might fail to import
class ComponentNotAvailable:
    """Placeholder for components that could not be loaded."""

    def __init__(self, name: str, error: Exception):
        self.name = name
        self.error = error

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(f"'{self.name}' is not available: {self.error}")


# Core components with smart import handling
try:
    from .core.client import OllamaClient
except ImportError as e:
    OllamaClient = cast(Any, ComponentNotAvailable("OllamaClient", e))

# Use placeholder for missing AsyncOllamaClient to avoid import error
try:
    # Import commented out until module is implemented
    # from .async_client import AsyncOllamaClient
    # Use placeholder until real implementation is available
    AsyncOllamaClient = cast(
        Any,
        ComponentNotAvailable(
            "AsyncOllamaClient", ImportError("Async client not yet implemented")
        ),
    )
except ImportError as e:
    AsyncOllamaClient = cast(Any, ComponentNotAvailable("AsyncOllamaClient", e))

# Dictionary to hold exception classes for proper typing and direct access
error_classes: Dict[str, Type[Exception]] = {}

# Load exception classes individually to avoid unused import warnings
try:
    # Import each exception class individually and store in our dictionary
    from .core.exceptions import (
        OllamaAPIError,  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    )

    error_classes["OllamaAPIError"] = OllamaAPIError

    from .core.exceptions import (
        OllamaConnectionError,  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    )

    error_classes["OllamaConnectionError"] = OllamaConnectionError

    from .core.exceptions import (
        OllamaModelNotFoundError,  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    )

    error_classes["OllamaModelNotFoundError"] = OllamaModelNotFoundError

    from .core.exceptions import (
        OllamaServerError,  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    )

    error_classes["OllamaServerError"] = OllamaServerError

    from .core.exceptions import (
        ConnectionError,  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    )

    error_classes["ConnectionError"] = ConnectionError

    from .core.exceptions import (
        TimeoutError,  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    )

    error_classes["TimeoutError"] = TimeoutError

    from .core.exceptions import (
        ModelNotFoundError,  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    )

    error_classes["ModelNotFoundError"] = ModelNotFoundError

    from .core.exceptions import (
        ServerError,  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    )

    error_classes["ServerError"] = ServerError

    from .core.exceptions import (
        InvalidRequestError,  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    )

    error_classes["InvalidRequestError"] = InvalidRequestError

    from .core.exceptions import (
        StreamingError,  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    )

    error_classes["StreamingError"] = StreamingError

    from .core.exceptions import (
        ParseError,  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    )

    error_classes["ParseError"] = ParseError
except ImportError as e:
    warnings.warn(f"Exception classes could not be imported: {e}")

# Re-export exceptions at module level for backwards compatibility
for name, cls in error_classes.items():
    globals()[name] = cls

# Define what's available when importing * from this package
# Explicitly list all exported symbols to avoid reportUnsupportedDunderAll
__all__ = [
    "OllamaClient",
    "AsyncOllamaClient",
    "__version__",
    "get_version_string",
    "get_default_api_url",
    "get_default_chat_model",
    "get_backup_chat_model",
    "get_default_embedding_model",
    "get_backup_embedding_model",
    # Explicitly list all error classes
    "OllamaAPIError",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaServerError",
    "ConnectionError",
    "TimeoutError",
    "ModelNotFoundError",
    "ServerError",
    "InvalidRequestError",
    "StreamingError",
    "ParseError",
]

# Debug mode detection
if os.environ.get("OLLAMA_FORGE_DEBUG") == "1":
    print(f"🔍 Ollama Forge v{__version__} loaded and ready")
    print(
        f"  - Client available: {'✅' if not isinstance(OllamaClient, ComponentNotAvailable) else '❌'}"
    )
    print(
        f"  - Async client available: {'✅' if not isinstance(AsyncOllamaClient, ComponentNotAvailable) else '❌'}"
    )
    print(f"  - Default model: {get_default_chat_model()}")
