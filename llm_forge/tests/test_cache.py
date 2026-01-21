import pytest
from llm_forge.caching.sqlite_cache import SQLiteCache
from llm_forge.core.interfaces import LLMResponse

def test_cache(tmp_path):
    db = tmp_path / "cache.db"
    cache = SQLiteCache(str(db))
    
    resp = LLMResponse(text="Hello", model_name="gpt-4", tokens_used=10)
    cache.set("key1", resp)
    
    cached = cache.get("key1")
    assert cached.text == "Hello"
    assert cached.meta["cached"] is True
    
    assert cache.get("key2") is None
