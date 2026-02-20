import pytest
from memory_forge.backends.chroma_store import ChromaBackend
from memory_forge.core.interfaces import MemoryItem


@pytest.fixture
def chroma_backend(tmp_path):
    return ChromaBackend("test_collection", str(tmp_path))


def test_chroma_crud(chroma_backend):
    item = MemoryItem("Test Content", embedding=[0.1] * 384, metadata={"tag": "A"})

    # Add
    chroma_backend.add(item)
    assert chroma_backend.count() == 1

    # Get
    retrieved = chroma_backend.get(item.id)
    assert retrieved.content == "Test Content"
    assert retrieved.metadata["tag"] == "A"

    # Search
    results = chroma_backend.search([0.1] * 384, limit=1)
    assert len(results) == 1

    # Delete
    chroma_backend.delete(item.id)
    assert chroma_backend.count() == 0


def test_chroma_no_embedding(chroma_backend):
    item = MemoryItem("No Embed")
    with pytest.raises(ValueError):
        chroma_backend.add(item)


def test_chroma_clear(chroma_backend):
    item = MemoryItem("A", embedding=[0.1] * 384)
    chroma_backend.add(item)
    chroma_backend.clear()
    assert chroma_backend.count() == 0
