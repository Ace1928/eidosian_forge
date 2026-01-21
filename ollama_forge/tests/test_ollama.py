import pytest
from ollama_forge import OllamaClient

def test_ollama_client_init():
    client = OllamaClient(base_url="http://mock-ollama:11434")
    assert client.base_url == "http://mock-ollama:11434"

# Note: Integration tests require a running Ollama instance, skipping real network calls here.
