from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generator
from pydantic import BaseModel, Field

class EngineConfig(BaseModel):
    """Configuration for local LLM engines."""
    model_path: str
    ctx_size: int = 4096
    threads: int = Field(default_factory=lambda: os.cpu_count() or 4)
    temp: float = 0.7
    n_predict: int = 1024
    extra_args: Dict[str, Any] = {}

class BaseEngine(ABC):
    """Abstract base class for all LLM engines."""

    def __init__(self, config: EngineConfig):
        self.config = config

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a complete response."""
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response tokens."""
        pass
