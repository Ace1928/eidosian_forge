from eidosian_core import eidosian
"""
Ollama Provider for LLM Forge.

Uses unified model configuration from eidos_mcp.config.models.
"""
from typing import List, Optional
from ..core.interfaces import LLMProvider, LLMResponse, EmbeddingProvider
from ollama_forge import OllamaClient
import httpx

# Import unified model configuration
try:
    from eidos_mcp.config.models import model_config
    UNIFIED_INFERENCE_MODEL = model_config.inference.model
    UNIFIED_EMBEDDING_MODEL = model_config.embedding.model
except ImportError:
    # Fallback if eidos_mcp not available
    UNIFIED_INFERENCE_MODEL = "phi3:mini"
    UNIFIED_EMBEDDING_MODEL = "nomic-embed-text"


class OllamaProvider(LLMProvider, EmbeddingProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        embedding_model: str = None,  # Uses unified config
        timeout: Optional[float] = None,
        default_model: str = None,  # Uses unified config
    ):
        self.client = OllamaClient(base_url=base_url, timeout=timeout)
        self.embedding_model = embedding_model or UNIFIED_EMBEDDING_MODEL
        self.default_model = default_model or UNIFIED_INFERENCE_MODEL

    @eidosian()
    def generate(self, prompt: str, model: str = None, timeout: Optional[float] = None, **kwargs) -> LLMResponse:
        actual_model = model if model else self.default_model
        resp = self.client.generate(actual_model, prompt, timeout=timeout, **kwargs)
        return LLMResponse(
            text=resp.response,
            tokens_used=0, 
            model_name=actual_model,
            meta={"duration": resp.total_duration}
        )

    @eidosian()
    def list_models(self) -> List[str]:
        return self.client.list_models()

    @eidosian()
    def embed_text(self, text: str) -> List[float]:
        """Embed a single string using Ollama."""
        # Manual call because OllamaClient might not have embed method yet
        # Or I should add it to OllamaClient. I'll add it here for now.
        url = f"{self.client.base_url}/api/embeddings"
        data = {
            "model": self.embedding_model,
            "prompt": text
        }
        with httpx.Client() as client:
            resp = client.post(url, json=data, timeout=self.client.timeout)
            resp.raise_for_status()
            return resp.json()["embedding"]

    @eidosian()
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]
