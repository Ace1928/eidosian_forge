from eidosian_core import eidosian

"""
Unified Model Configuration for Eidosian Forge.

This module provides centralized configuration for all LLM and embedding
models used across the Eidosian Forge ecosystem. All forges should import
their model settings from here to ensure consistency.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │           EIDOS MODEL CONFIGURATION                  │
    ├─────────────────────────────────────────────────────┤
    │  Inference: phi3:mini (2.2GB, best reasoning)       │
    │  Embedding: nomic-embed-text (768d, 8192 context)   │
    │  Fast Embed: all-minilm (384d, 512 context)         │
    └─────────────────────────────────────────────────────┘
    
Usage:
    from eidos_mcp.config.models import ModelConfig
    
    config = ModelConfig()
    embed = config.get_embedding(text)
    response = config.generate(prompt)
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from eidosian_core.ports import get_service_url

from .. import FORGE_ROOT

# Configuration paths
CONFIG_DIR = Path(os.environ.get("EIDOS_MODEL_CONFIG_DIR", str(FORGE_ROOT / "data"))).resolve()
CONFIG_FILE = CONFIG_DIR / "model_config.json"
DEFAULT_OLLAMA_BASE_URL = os.environ.get(
    "EIDOS_OLLAMA_BASE_URL",
    get_service_url("ollama_http", default_port=11434, default_host="localhost", default_path=""),
).rstrip("/")
DEFAULT_OLLAMA_API_V1_URL = f"{DEFAULT_OLLAMA_BASE_URL}/v1"


@dataclass
class OllamaConfig:
    """Ollama server configuration."""

    base_url: str = DEFAULT_OLLAMA_BASE_URL
    api_v1_url: str = DEFAULT_OLLAMA_API_V1_URL
    timeout: float = 3600.0  # [EIDOS] 1 hour for slow local models


@dataclass
class InferenceConfig:
    """Inference model configuration."""

    model: str = "phi3:mini"
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model: str = "nomic-embed-text"
    dimensions: int = 768
    context_length: int = 8192


@dataclass
class FastEmbeddingConfig:
    """Fast/lightweight embedding configuration."""

    model: str = "all-minilm"
    dimensions: int = 384
    context_length: int = 512


class ModelConfig:
    """
    Unified model configuration for the Eidosian Forge ecosystem.

    This class provides:
    - Centralized model configuration
    - Embedding generation via Ollama
    - Text generation via Ollama
    - Caching for repeated embeddings

    All forges should use this for model operations to ensure consistency.
    """

    _instance: Optional["ModelConfig"] = None

    def __new__(cls) -> "ModelConfig":
        """Singleton pattern for consistent configuration."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._load_config()
        self._embedding_cache: Dict[str, List[float]] = {}

    def _load_config(self):
        """Load configuration from JSON file or use defaults."""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                data = json.load(f)

            self.ollama = OllamaConfig(
                base_url=data.get("ollama", {}).get("base_url", DEFAULT_OLLAMA_BASE_URL),
                api_v1_url=data.get("ollama", {}).get("api_v1_url", DEFAULT_OLLAMA_API_V1_URL),
            )
            self.inference = InferenceConfig(
                model=data.get("inference", {}).get("model", "phi3:mini"),
                max_tokens=data.get("inference", {}).get("max_tokens", 4096),
                temperature=data.get("inference", {}).get("temperature", 0.7),
            )
            self.embedding = EmbeddingConfig(
                model=data.get("embedding", {}).get("model", "nomic-embed-text"),
                dimensions=data.get("embedding", {}).get("dimensions", 768),
                context_length=data.get("embedding", {}).get("context_length", 8192),
            )
            self.fast_embedding = FastEmbeddingConfig(
                model=data.get("fast_embedding", {}).get("model", "all-minilm"),
                dimensions=data.get("fast_embedding", {}).get("dimensions", 384),
                context_length=data.get("fast_embedding", {}).get("context_length", 512),
            )
        else:
            # Use defaults
            self.ollama = OllamaConfig()
            self.inference = InferenceConfig()
            self.embedding = EmbeddingConfig()
            self.fast_embedding = FastEmbeddingConfig()

    @eidosian()
    def embed_text(self, text: str, model: Optional[str] = None, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: The text to embed
            model: Override model (default: nomic-embed-text)
            use_cache: Whether to use cached embeddings

        Returns:
            List of floats representing the embedding vector
        """
        actual_model = model or self.embedding.model
        cache_key = f"{actual_model}:{text[:100]}"  # Truncate for cache key

        if use_cache and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        url = f"{self.ollama.base_url}/api/embeddings"
        data = {"model": actual_model, "prompt": text}

        with httpx.Client(timeout=self.ollama.timeout) as client:
            resp = client.post(url, json=data)
            resp.raise_for_status()
            embedding = resp.json()["embedding"]

        if use_cache:
            self._embedding_cache[cache_key] = embedding

        return embedding

    @eidosian()
    def embed_batch(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Override model

        Returns:
            List of embedding vectors
        """
        return [self.embed_text(t, model=model) for t in texts]

    @eidosian()
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate text using the inference model.

        Args:
            prompt: The prompt to generate from
            model: Override model (default: phi3:mini)
            max_tokens: Override max tokens
            temperature: Override temperature
            system: System prompt
            **kwargs: Additional Ollama parameters

        Returns:
            Generated text string
        """
        actual_model = model or self.inference.model
        actual_max_tokens = max_tokens or self.inference.max_tokens
        actual_temp = temperature if temperature is not None else self.inference.temperature

        url = f"{self.ollama.base_url}/api/generate"
        data = {
            "model": actual_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": actual_max_tokens,
                "temperature": actual_temp,
            },
            **kwargs,
        }

        if system:
            data["system"] = system

        with httpx.Client(timeout=self.ollama.timeout) as client:
            resp = client.post(url, json=data)
            resp.raise_for_status()
            return resp.json()["response"]

    @eidosian()
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs) -> str:
        """
        Chat completion using the inference model.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            model: Override model
            **kwargs: Additional parameters

        Returns:
            Assistant's response text
        """
        actual_model = model or self.inference.model

        url = f"{self.ollama.base_url}/api/chat"
        data = {"model": actual_model, "messages": messages, "stream": False, **kwargs}

        with httpx.Client(timeout=self.ollama.timeout) as client:
            resp = client.post(url, json=data)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    @eidosian()
    def is_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.ollama.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    @eidosian()
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.ollama.base_url}/api/tags")
                if resp.status_code == 200:
                    return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    @eidosian()
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return {
            "ollama": {
                "base_url": self.ollama.base_url,
                "api_v1_url": self.ollama.api_v1_url,
            },
            "inference": {
                "model": self.inference.model,
                "max_tokens": self.inference.max_tokens,
                "temperature": self.inference.temperature,
            },
            "embedding": {
                "model": self.embedding.model,
                "dimensions": self.embedding.dimensions,
                "context_length": self.embedding.context_length,
            },
            "fast_embedding": {
                "model": self.fast_embedding.model,
                "dimensions": self.fast_embedding.dimensions,
                "context_length": self.fast_embedding.context_length,
            },
        }


# Singleton instance for easy import
model_config = ModelConfig()


@eidosian()
def get_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """Convenience function for embedding text."""
    return model_config.embed_text(text, model=model)


@eidosian()
def generate(prompt: str, **kwargs) -> str:
    """Convenience function for text generation."""
    return model_config.generate(prompt, **kwargs)


@eidosian()
def chat(messages: List[Dict[str, str]], **kwargs) -> str:
    """Convenience function for chat completion."""
    return model_config.chat(messages, **kwargs)


# Export all
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
