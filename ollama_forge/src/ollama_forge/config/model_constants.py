"""
Model Constants for Ollama Forge

This module defines all model-related constants and provides utility functions for
model selection, fallback mechanisms, and alias resolution.

Exemplifying Eidosian principles:
- Contextual Integrity: Each constant serves a clear purpose
- Structure as Control: Organized hierarchically by function
- Precision as Style: Exact naming and clear documentation
"""

from typing import Dict

# Standard model types
MODEL_TYPE_CHAT = "chat"
MODEL_TYPE_COMPLETION = "completion"
MODEL_TYPE_EMBEDDING = "embedding"

# Recommended models by category - optimized for different use cases
RECOMMENDED_MODELS = {
    # Fast, small models for quick responses
    "small": {
        MODEL_TYPE_CHAT: ["qwen2.5:0.5b-instruct", "deepseek-r1:1.5b"],
        MODEL_TYPE_COMPLETION: ["qwen2.5:0.5b", "deepseek-r1:1.5b"],
        MODEL_TYPE_EMBEDDING: ["qwen2.5:0.5b", "deepseek-r1:1.5b"],
    },
    # Medium models balancing quality and performance
    "medium": {
        MODEL_TYPE_CHAT: ["qwen2.5:7b-instruct", "deepseek-r1:7b"],
        MODEL_TYPE_COMPLETION: ["qwen2.5:7b", "deepseek-r1:7b"],
        MODEL_TYPE_EMBEDDING: ["qwen2.5:0.5b", "deepseek-r1:1.5b"],
    },
}

# Default and backup models - balanced for general use
DEFAULT_CHAT_MODEL = "deepseek-r1:1.5b"
BACKUP_CHAT_MODEL = "qwen2.5:0.5b-Instruct"
DEFAULT_EMBEDDING_MODEL = (
    DEFAULT_CHAT_MODEL  # Using same model improves semantic consistency
)
BACKUP_EMBEDDING_MODEL = BACKUP_CHAT_MODEL

# Model aliases for convenience
MODEL_ALIASES: Dict[str, str] = {
    # Generics
    "default": DEFAULT_CHAT_MODEL,
    "small": "deepseek-r1:1.5b",
    "medium": "qwen2.5:7b-instruct",
    # Specific types
    "embed": "nomic-embed-text",
    "embedding": "nomic-embed-text",
    "code": "codellama:7b",
    "python": "codellama:7b-python",
    # Name variations
    "deepseek-small": "deepseek-r1:1.5b",
    "deepseek-medium": "deepseek-r1:7b",
    "deepseek-large": "deepseek-r1:32b",
    "qwen-small": "qwen2.5:0.5b-instruct",
    "qwen-medium": "qwen2.5:7b-instruct",
    "qwen-large": "qwen2.5:32b-instruct",
}


def resolve_model_alias(model_name: str) -> str:
    """
    Resolve a model alias to its actual model name.

    Args:
        model_name: The model name or alias

    Returns:
        Resolved model name
    """
    return MODEL_ALIASES.get(model_name.lower(), model_name)


def get_fallback_model(model_name: str, model_type: str = MODEL_TYPE_CHAT) -> str:
    """
    Get an appropriate fallback model if the requested one is unavailable.

    Args:
        model_name: The originally requested model name
        model_type: The type of model (chat, completion, embedding)

    Returns:
        Name of an appropriate fallback model
    """
    # Start with default fallbacks based on type
    if model_type == MODEL_TYPE_EMBEDDING:
        default_fallback = BACKUP_EMBEDDING_MODEL
    else:
        default_fallback = BACKUP_CHAT_MODEL

    # Try to find a model of similar size/capability
    for _, models in RECOMMENDED_MODELS.items():
        if model_type in models and any(
            model_name.startswith(m.split(":")[0]) for m in models[model_type]
        ):
            # Return the first recommended model for this size/type
            return models[model_type][0]

    return default_fallback


def get_model_recommendation(task: str, size: str = "medium") -> str:
    """
    Get a model recommendation for a specific task and size preference.

    Args:
        task: Description of the task ("chat", "code", "embedding", etc.)
        size: Desired model size ("small", "medium", "large")

    Returns:
        Recommended model name
    """
    # Map task to model type
    task_to_type = {
        "chat": MODEL_TYPE_CHAT,
        "conversation": MODEL_TYPE_CHAT,
        "completion": MODEL_TYPE_COMPLETION,
        "generate": MODEL_TYPE_COMPLETION,
        "embedding": MODEL_TYPE_EMBEDDING,
        "embed": MODEL_TYPE_EMBEDDING,
        "code": MODEL_TYPE_COMPLETION,
        "programming": MODEL_TYPE_COMPLETION,
    }

    # Handle task-specific recommendations
    if task.lower() == "code" or task.lower() == "programming":
        return "codellama:7b-python" if size != "small" else "deepseek-r1:1.5b"

    model_type = task_to_type.get(task.lower(), MODEL_TYPE_CHAT)
    size = size.lower() if size.lower() in RECOMMENDED_MODELS else "medium"

    if model_type in RECOMMENDED_MODELS[size]:
        return RECOMMENDED_MODELS[size][model_type][0]

    # Default fallbacks
    if model_type == MODEL_TYPE_EMBEDDING:
        return DEFAULT_EMBEDDING_MODEL
    return DEFAULT_CHAT_MODEL
