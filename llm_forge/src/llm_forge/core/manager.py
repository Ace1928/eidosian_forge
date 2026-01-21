from typing import Dict, Optional
from .interfaces import LLMProvider, LLMResponse
from ..providers.openai_provider import OpenAIProvider

class ModelManager:
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}

    def register_provider(self, name: str, provider: LLMProvider):
        self.providers[name] = provider

    def generate(self, prompt: str, provider_name: str, **kwargs) -> LLMResponse:
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not found")
        return self.providers[provider_name].generate(prompt, **kwargs)