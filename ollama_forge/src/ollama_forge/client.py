"""
Ollama Forge - Python Client.
"""
import httpx
from typing import Dict, Any, Generator, List, Optional
from pydantic import BaseModel

class OllamaResponse(BaseModel):
    model: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None

class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: Optional[float] = None,
    ):
        self.base_url = base_url
        self.timeout = timeout

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> OllamaResponse:
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        
        # Use provided timeout or default client timeout
        current_timeout = self.timeout if timeout is None else timeout
        
        with httpx.Client() as client:
            resp = client.post(url, json=data, timeout=current_timeout)
            resp.raise_for_status()
            
            if stream:
                raise NotImplementedError("Streaming not supported in simple client yet")
            
            return OllamaResponse(**resp.json())

    def list_models(self) -> List[str]:
        url = f"{self.base_url}/api/tags"
        with httpx.Client() as client:
            resp = client.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            return [m["name"] for m in resp.json().get("models", [])]
