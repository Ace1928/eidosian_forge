"""
Eidos MCP Configuration Package.

Provides unified configuration for all Eidosian Forge components.
"""

from .models import (
    EmbeddingConfig,
    FastEmbeddingConfig,
    InferenceConfig,
    ModelConfig,
    OllamaConfig,
    chat,
    generate,
    get_embedding,
    model_config,
)

__all__ = [
    "ModelConfig",
    "model_config",
    "get_embedding",
    "generate",
    "chat",
    "OllamaConfig",
    "InferenceConfig",
    "EmbeddingConfig",
    "FastEmbeddingConfig",
]
