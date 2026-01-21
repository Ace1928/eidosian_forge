import pytest
from unittest.mock import MagicMock, patch
from llm_forge.providers.openai_provider import OpenAIProvider
from llm_forge.providers.ollama_provider import OllamaProvider

@patch("openai.Client")
def test_openai_provider(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    # Mock completion
    mock_choice = MagicMock()
    mock_choice.message.content = "AI Response"
    mock_client.chat.completions.create.return_value.choices = [mock_choice]
    mock_client.chat.completions.create.return_value.usage.total_tokens = 10
    
    # Mock embeddings
    mock_embed = MagicMock()
    mock_embed.embedding = [0.1, 0.2]
    mock_client.embeddings.create.return_value.data = [mock_embed]
    
    provider = OpenAIProvider("fake-key")
    resp = provider.generate("Prompt")
    assert resp.text == "AI Response"
    
    emb = provider.embed_text("Text")
    assert emb == [0.1, 0.2]

@patch("llm_forge.providers.ollama_provider.OllamaClient")
def test_ollama_provider(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    mock_resp = MagicMock()
    mock_resp.response = "Llama Response"
    mock_client.generate.return_value = mock_resp
    mock_client.list_models.return_value = ["llama2"]
    
    provider = OllamaProvider()
    resp = provider.generate("Prompt")
    assert resp.text == "Llama Response"
    
    models = provider.list_models()
    assert "llama2" in models
