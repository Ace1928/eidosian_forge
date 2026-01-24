"""
Eidos MCP Configuration Package.

Provides unified configuration for all Eidosian Forge components.
"""

from .models import (
    ModelConfig,
    model_config,
    get_embedding,
    generate,
    chat,
    OllamaConfig,
    InferenceConfig,
    EmbeddingConfig,
    FastEmbeddingConfig,
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
