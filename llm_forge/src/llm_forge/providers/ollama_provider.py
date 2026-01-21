from typing import List
from ..core.interfaces import LLMProvider, LLMResponse, EmbeddingProvider
from ollama_forge import OllamaClient
import httpx

class OllamaProvider(LLMProvider, EmbeddingProvider):
    def __init__(self, base_url: str = "http://localhost:11434", embedding_model: str = "nomic-embed-text"):
        self.client = OllamaClient(base_url=base_url)
        self.embedding_model = embedding_model

    def generate(self, prompt: str, model: str = "qwen2.5:0.5b", **kwargs) -> LLMResponse:
        resp = self.client.generate(model, prompt, **kwargs)
        return LLMResponse(
            text=resp.response,
            tokens_used=0, 
            model_name=model,
            meta={"duration": resp.total_duration}
        )

    def list_models(self) -> List[str]:
        return self.client.list_models()

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
            resp = client.post(url, json=data, timeout=30.0)
            resp.raise_for_status()
            return resp.json()["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]