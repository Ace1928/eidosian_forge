"""Utility types and functions for the Ollama Forge framework.

This module provides core type definitions, protocols, and utility functions
that form the foundation of type-safe operations throughout the Ollama Forge
framework. Import these definitions to ensure consistent typing across
client implementations and model management operations.

Type hierarchy follows a "define once, use everywhere" philosophy to maintain
structural coherence across the system.
"""

from typing import List

# Re-export type definitions for cleaner imports
from ollama_forge.utils.type_definitions import (  # Core model representation; Protocols defining capabilities; API response types
    ChatResponse,
    DownloadProgress,
    EmbeddingResponse,
    GenerateResponse,
    MessageContent,
    ModelInfo,
    ModelInstaller,
    ModelProvider,
    ModelSize,
    ModelSource,
    ModelsResponse,
    PullProgress,
    VersionResponse,
)

__all__: List[str] = [
    # Core model representation
    "ModelInfo",
    "ModelSize",
    "ModelSource",
    # Protocols defining capabilities
    "DownloadProgress",
    "ModelInstaller",
    "ModelProvider",
    # API response types
    "ChatResponse",
    "EmbeddingResponse",
    "GenerateResponse",
    "MessageContent",
    "ModelsResponse",
    "PullProgress",
    "VersionResponse",
]
