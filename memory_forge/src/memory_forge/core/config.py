"""
Configuration logic for Memory Forge.
"""
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Literal

class BackendConfig(BaseModel):
    type: Literal["chroma", "sqlite", "json", "postgres"] = "chroma"
    connection_string: str = "./data/memory.db"
    collection_name: str = "eidos_memory"

class MemoryConfig(BaseModel):
    episodic: BackendConfig = Field(default_factory=lambda: BackendConfig(type="chroma", collection_name="episodic"))
    semantic: BackendConfig = Field(default_factory=lambda: BackendConfig(type="chroma", collection_name="semantic"))
    embedding_model: str = "all-MiniLM-L6-v2"
