"""
Configuration logic for Memory Forge.
"""
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Literal

class BackendConfig(BaseModel):
    type: Literal["chroma", "sqlite", "json", "postgres"] = "json"
    connection_string: str = "memory_forge_episodic.json"
    collection_name: str = "eidos_memory"

class MemoryConfig(BaseModel):
    episodic: BackendConfig = Field(default_factory=lambda: BackendConfig(type="json", connection_string="./data/episodic_memory.json", collection_name="episodic"))
    semantic: BackendConfig = Field(default_factory=lambda: BackendConfig(type="json", connection_string="./data/semantic_memory.json", collection_name="semantic"))
    embedding_model: str = "all-MiniLM-L6-v2"
