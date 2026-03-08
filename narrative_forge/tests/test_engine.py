import pytest
from narrative_forge.engine import NarrativeEngine
from ollama_forge import OllamaClient

def test_narrative_engine_real():
    # Check if model available
    client = OllamaClient()
    try:
        models = client.list_models()
    except Exception as exc:
        pytest.skip(f"Ollama server unavailable: {exc}")
    if "qwen3.5:2b" not in models:
        pytest.skip("Model qwen3.5:2b not found")

    engine = NarrativeEngine(
        memory_path=":memory:", 
        think_interval=100,
        provider="ollama",
        model_name="qwen3.5:2b"
    )
    
    resp = engine.respond("Say 'I am Eidos'")
    
    assert len(resp) > 0
    assert len(engine.store.data.interactions) == 1
    engine.shutdown()
