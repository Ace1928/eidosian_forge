from eidosian_core import eidosian
"""
LLM Forge - Model Manager.

Uses unified model configuration from eidos_mcp.config.models.
"""
import os
from typing import Dict, Optional, Any
from .interfaces import LLMProvider, LLMResponse
from ..providers.openai_provider import OpenAIProvider
from ..providers.ollama_provider import OllamaProvider

# Import unified model configuration
try:
    from eidos_mcp.config.models import model_config
    DEFAULT_INFERENCE_MODEL = model_config.inference.model
    DEFAULT_OLLAMA_URL = model_config.ollama.base_url
except ImportError:
    DEFAULT_INFERENCE_MODEL = "phi3:mini"
    DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _parse_timeout(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if cleaned in {"", "none", "null", "inf", "infinite"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


class ModelManager:
    """
    Unified model manager for LLM operations.
    
    Uses centralized configuration from eidos_mcp.config.models
    to ensure consistent model usage across all forges.
    """
    
    def __init__(
        self,
        default_ollama_url: str = None,
        default_ollama_timeout: Optional[float] = None,
        default_ollama_model: str = None,
    ):
        # Use unified config defaults
        default_ollama_url = default_ollama_url or DEFAULT_OLLAMA_URL
        default_ollama_model = default_ollama_model or DEFAULT_INFERENCE_MODEL
        
        env_timeout = _parse_timeout(os.environ.get("EIDOS_OLLAMA_TIMEOUT"))
        if env_timeout is not None:
            default_ollama_timeout = env_timeout
        self.providers: Dict[str, LLMProvider] = {}
        self.ollama_provider = OllamaProvider(
            base_url=default_ollama_url,
            timeout=default_ollama_timeout,
            default_model=default_ollama_model,
        )
        self.register_provider("ollama", self.ollama_provider)
        # Optionally register OpenAIProvider if API key is present
        if OpenAIProvider.is_available():
            self.register_provider("openai", OpenAIProvider())

    @eidosian()
    def register_provider(self, name: str, provider: LLMProvider):
        self.providers[name] = provider

    @eidosian()
    def generate(self, prompt: str, provider_name: str = "ollama", model: Optional[str] = None, timeout: Optional[float] = None, **kwargs: Any) -> LLMResponse:
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not found")
        
        # Pass model and timeout if provided, otherwise let provider use its defaults
        gen_kwargs = {"model": model, "timeout": timeout, **kwargs}
        # Filter out None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        return self.providers[provider_name].generate(prompt, **gen_kwargs)
