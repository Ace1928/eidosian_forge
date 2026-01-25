from eidosian_core import eidosian
"""
Ollama Forge - Python Client.

[EIDOS] Configured for local LLM with extended timeouts.
"""
import httpx
from typing import Dict, Any, Generator, List, Optional
from pydantic import BaseModel

# [EIDOS] Default timeout for LLM operations - 1 hour for slow local models
DEFAULT_LLM_TIMEOUT = 3600.0

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
        timeout: Optional[float] = DEFAULT_LLM_TIMEOUT,
    ):
        self.base_url = base_url
        self.timeout = timeout

    @eidosian()
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

    @eidosian()
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send a chat request to the Ollama API.
        
        Args:
            model: Name of the model to use
            messages: List of message dictionaries with 'role' and 'content' keys
            stream: Whether to stream the response (not yet implemented)
            options: Additional options like temperature, max_tokens
            timeout: Request timeout in seconds
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response containing the model's reply
        """
        url = f"{self.base_url}/api/chat"
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        if options:
            data["options"] = options
        
        # Use provided timeout or default client timeout
        current_timeout = self.timeout if timeout is None else timeout
        
        with httpx.Client() as client:
            resp = client.post(url, json=data, timeout=current_timeout)
            resp.raise_for_status()
            
            if stream:
                raise NotImplementedError("Streaming not supported in simple client yet")
            
            return resp.json()

    @eidosian()
    def list_models(self) -> List[str]:
        """List all available models."""
        url = f"{self.base_url}/api/tags"
        with httpx.Client() as client:
            resp = client.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            return [m["name"] for m in resp.json().get("models", [])]

    @eidosian()
    def get_version(self) -> Dict[str, Any]:
        """Get Ollama server version information."""
        url = f"{self.base_url}/api/version"
        with httpx.Client() as client:
            resp = client.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()

    @eidosian()
    def delete_model(self, model: str) -> bool:
        """Delete a model from the Ollama server."""
        url = f"{self.base_url}/api/delete"
        data = {"model": model}
        with httpx.Client() as client:
            resp = client.request("DELETE", url, json=data, timeout=self.timeout)
            return resp.status_code == 200

    @eidosian()
    def create_embedding(
        self,
        model: str,
        prompt: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Create embeddings for the given text.
        
        Args:
            model: Name of the embedding model
            prompt: Text to create embeddings for
            timeout: Request timeout
            
        Returns:
            Dictionary containing embedding vector
        """
        url = f"{self.base_url}/api/embed"
        data = {"model": model, "input": prompt}
        current_timeout = self.timeout if timeout is None else timeout
        
        with httpx.Client() as client:
            resp = client.post(url, json=data, timeout=current_timeout)
            resp.raise_for_status()
            return resp.json()

    @eidosian()
    def batch_embeddings(
        self,
        model: str,
        prompts: List[str],
        timeout: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            model: Name of the embedding model
            prompts: List of texts to create embeddings for
            timeout: Request timeout
            
        Returns:
            List of embedding results
        """
        results = []
        for prompt in prompts:
            result = self.create_embedding(model=model, prompt=prompt, timeout=timeout)
            results.append(result)
        return results

    @eidosian()
    def pull_model(self, model: str, stream: bool = False) -> Dict[str, Any]:
        """Pull a model from the Ollama library."""
        url = f"{self.base_url}/api/pull"
        data = {"model": model, "stream": stream}
        with httpx.Client() as client:
            resp = client.post(url, json=data, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()

    @eidosian()
    def show_model(self, model: str) -> Dict[str, Any]:
        """Show information about a model."""
        url = f"{self.base_url}/api/show"
        data = {"model": model}
        with httpx.Client() as client:
            resp = client.post(url, json=data, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
