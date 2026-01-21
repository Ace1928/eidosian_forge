import pytest
from llm_forge import ModelManager, LLMResponse, LLMProvider

class MockProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(text="Mock Response", model_name="mock")
    
    def list_models(self):
        return ["mock"]

def test_manager():
    mm = ModelManager()
    mm.register_provider("mock", MockProvider())
    
    resp = mm.generate("Hi", "mock")
    assert resp.text == "Mock Response"