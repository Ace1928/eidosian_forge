import os
import openai
from typing import List, Dict, Any, Optional
from ..core.interfaces import LLMProvider, LLMResponse, EmbeddingProvider
from eidosian_core import eidosian

class OpenAIProvider(LLMProvider, EmbeddingProvider):
    @staticmethod
    def is_available() -> bool:
        return "OPENAI_API_KEY" in os.environ

    def __init__(self, api_key: Optional[str] = None, base_url: str = None, embedding_model: str = "text-embedding-3-small"):
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables.")
        self.client = openai.Client(api_key=api_key, base_url=base_url)
        self.embedding_model = embedding_model

    @eidosian()
    def generate(self, prompt: str, model: str = "gpt-3.5-turbo", **kwargs) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return LLMResponse(
            text=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            model_name=model
        )

    @eidosian()
    def list_models(self) -> List[str]:
        return [m.id for m in self.client.models.list().data]

    @eidosian()
    def embed_text(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.embedding_model).data[0].embedding

    @eidosian()
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # OpenAI handles batching automatically up to a limit
        texts = [t.replace("\n", " ") for t in texts]
        data = self.client.embeddings.create(input=texts, model=self.embedding_model).data
        return [d.embedding for d in data]
