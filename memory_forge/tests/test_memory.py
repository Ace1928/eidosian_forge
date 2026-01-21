import pytest
import shutil
from pathlib import Path
from memory_forge import MemoryForge, MemoryConfig

def test_memory_forge_flow(tmp_path):
    # Setup Config using JSON backend for speed/simplicity in tests
    db_path = tmp_path / "mem.json"
    config = MemoryConfig()
    config.episodic.type = "json"
    config.episodic.connection_string = str(db_path)
    
    mf = MemoryForge(config)
    
    # Fake embedding
    vec = [0.1, 0.2, 0.3]
    
    # Remember
    mid = mf.remember("I ate a pizza", embedding=vec)
    assert mid is not None
    
    # Stats
    assert mf.stats()["episodic_count"] == 1
    
    # Recall
    results = mf.recall(vec, limit=1)
    assert len(results) == 1
    assert results[0].content == "I ate a pizza"