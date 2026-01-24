"""
Configuration logic for Memory Forge.

Uses unified model configuration from eidos_mcp.config.models.
"""
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Literal, Optional

# Import unified model configuration
try:
    from eidos_mcp.config.models import model_config
    UNIFIED_EMBEDDING_MODEL = model_config.embedding.model
    EMBEDDING_PROVIDER = "ollama"  # Use Ollama instead of HuggingFace
except ImportError:
    UNIFIED_EMBEDDING_MODEL = "nomic-embed-text"
    EMBEDDING_PROVIDER = "ollama"


class BackendConfig(BaseModel):
    type: Literal["chroma", "sqlite", "json", "postgres"] = "json"
    connection_string: str = "memory_forge_episodic.json"
    collection_name: str = "eidos_memory"


class MemoryConfig(BaseModel):
    episodic: BackendConfig = Field(default_factory=lambda: BackendConfig(type="json", connection_string="./data/episodic_memory.json", collection_name="episodic"))
    semantic: BackendConfig = Field(default_factory=lambda: BackendConfig(type="json", connection_string="./data/semantic_memory.json", collection_name="semantic"))
    embedding_model: str = UNIFIED_EMBEDDING_MODEL
    embedding_provider: str = EMBEDDING_PROVIDER  # "ollama" or "huggingface"
