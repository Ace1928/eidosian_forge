import pytest
from llm_forge.providers.ollama_provider import OllamaProvider

def test_ollama_real_generation():
    provider = OllamaProvider()
    
    # Check if model exists
    try:
        models = provider.list_models()
    except Exception as exc:
        pytest.skip(f"Ollama server unavailable: {exc}")
    model = "qwen2.5:0.5b"
    if model not in models:
        pytest.skip(f"Model {model} not found locally")
        
    resp = provider.generate("Say hello!", model=model)
    assert len(resp.text) > 0
    assert resp.model_name == model

def test_ollama_real_embedding():
    provider = OllamaProvider(embedding_model="nomic-embed-text")
    
    try:
        models = provider.list_models()
    except Exception as exc:
        pytest.skip(f"Ollama server unavailable: {exc}")
    if "nomic-embed-text:latest" not in models and "nomic-embed-text" not in models:
         pytest.skip("nomic-embed-text not found")
         
    vec = provider.embed_text("Eidos")
    assert len(vec) > 0
    assert isinstance(vec[0], float)
