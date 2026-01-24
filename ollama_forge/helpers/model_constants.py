"""Model constants for ollama_forge.

Uses unified model configuration from eidos_mcp when available,
falls back to sensible defaults otherwise.
"""

try:
    from eidos_mcp.config.models import get_model_config
    config = get_model_config()
    DEFAULT_MODEL = config.inference_model
    DEFAULT_EMBEDDING_MODEL = config.embedding_model
except ImportError:
    DEFAULT_MODEL = "phi3:mini"
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Chat model aliases
DEFAULT_CHAT_MODEL = DEFAULT_MODEL
BACKUP_CHAT_MODEL = "llama3.2:latest"

# Backup models for fallback
BACKUP_MODEL = "llama3.2:latest"
BACKUP_EMBEDDING_MODEL = "all-minilm"


def get_fallback_model(preferred: str = None) -> str:
    """Get a fallback model if preferred is not available."""
    from ollama_forge import OllamaClient
    try:
        client = OllamaClient()
        models = client.list_models()
        if preferred and preferred in models:
            return preferred
        if DEFAULT_MODEL in models:
            return DEFAULT_MODEL
        if BACKUP_MODEL in models:
            return BACKUP_MODEL
        if models:
            return models[0]
    except Exception:
        pass
    return DEFAULT_MODEL

