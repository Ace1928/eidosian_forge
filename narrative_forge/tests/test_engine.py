import pytest
from narrative_forge.engine import NarrativeEngine
from ollama_forge import OllamaClient

def test_narrative_engine_real():
    # Check if model available
    client = OllamaClient()
    models = client.list_models()
    if "qwen2.5:0.5b" not in models:
        pytest.skip("Model qwen2.5:0.5b not found")

    engine = NarrativeEngine(
        memory_path=":memory:", 
        think_interval=100,
        provider="ollama",
        model_name="qwen2.5:0.5b"
    )
    
    resp = engine.respond("Say 'I am Eidos'")
    
    assert len(resp) > 0
    assert len(engine.store.data.interactions) == 1
    engine.shutdown()
