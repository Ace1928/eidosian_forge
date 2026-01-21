import pytest
import shutil
import json
from pathlib import Path
from memory_forge.backends.json_store import JsonBackend
from memory_forge.core.interfaces import MemoryItem

@pytest.fixture
def json_backend(tmp_path):
    return JsonBackend(str(tmp_path / "db.json"))

def test_json_crud(json_backend):
    item = MemoryItem("Test Content", embedding=[0.1, 0.2])
    
    # Add
    assert json_backend.add(item)
    assert json_backend.count() == 1
    
    # Get
    retrieved = json_backend.get(item.id)
    assert retrieved.content == "Test Content"
    
    # Search
    results = json_backend.search([0.1, 0.2], limit=1)
    assert len(results) == 1
    assert results[0].id == item.id
    
    # Delete
    assert json_backend.delete(item.id)
    assert json_backend.count() == 0
    assert json_backend.get(item.id) is None

def test_json_search_empty(json_backend):
    assert json_backend.search([0.1]) == []

def test_json_delete_nonexistent(json_backend):
    assert not json_backend.delete("fake")

def test_json_corrupt_file(tmp_path):
    p = tmp_path / "corrupt.json"
    p.write_text("{invalid_json")
    jb = JsonBackend(str(p))
    assert jb.count() == 0 # Should handle gracefully
