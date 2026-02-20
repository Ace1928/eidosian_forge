"""Integration tests for word_forge.vectorizer.vector_store."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


# Skip all tests in this module if vector dependencies are unavailable
def _vector_available() -> bool:
    if importlib.util.find_spec("chromadb") is None or importlib.util.find_spec("sentence_transformers") is None:
        return False
    try:
        import word_forge.vectorizer.vector_store as vector_store
    except Exception:
        return False
    return vector_store.SentenceTransformer is not None


_VECTOR_AVAILABLE = _vector_available()

pytestmark = pytest.mark.skipif(
    not _VECTOR_AVAILABLE,
    reason="Vector dependencies (chromadb, sentence-transformers) not installed",
)

from word_forge.configs.config_essentials import StorageType
from word_forge.vectorizer.vector_store import (
    InitializationError,
    SearchError,
    VectorStore,
)

TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _sample_entry() -> dict:
    return {
        "id": "1",
        "id_int": 1,
        "term": "happy",
        "definition": "feeling or showing pleasure",
        "part_of_speech": "adj",
        "usage_examples": ["I feel happy today."],
        "language": "en",
        "last_refreshed": 0.0,
        "relationships": [],
    }


class TestVectorStoreDemoMode:
    def test_demo_mode_requires_explicit_flag(self) -> None:
        with pytest.raises(InitializationError):
            VectorStore(
                dimension=384,
                storage_type=StorageType.MEMORY,
                model_name=TEST_MODEL,
            )

    def test_demo_mode_initialization(self) -> None:
        store = VectorStore(
            dimension=384,
            storage_type=StorageType.MEMORY,
            demo_mode=True,
            model_name=TEST_MODEL,
        )
        assert store.demo_mode is True
        assert store.backend_name == "memory-demo"


class TestVectorStoreBehavior:
    def test_search_requires_input(self) -> None:
        store = VectorStore(
            dimension=384,
            storage_type=StorageType.MEMORY,
            demo_mode=True,
            model_name=TEST_MODEL,
        )
        with pytest.raises(SearchError):
            store.search()

    def test_store_word_in_memory(self) -> None:
        store = VectorStore(
            dimension=384,
            storage_type=StorageType.MEMORY,
            demo_mode=True,
            model_name=TEST_MODEL,
        )
        stored = store.store_word(_sample_entry())
        assert stored > 0
        assert store.collection.count() > 0

    def test_upsert_normalizes_mismatched_dimension(self) -> None:
        store = VectorStore(
            dimension=1024,
            storage_type=StorageType.MEMORY,
            demo_mode=True,
            model_name=TEST_MODEL,
        )
        small_vec = np.ones(384, dtype=np.float32)
        store.upsert("abc", small_vec, metadata={"content_type": "word"})
        assert store.collection.count() == 1
        stored = store.collection._store["abc"]["embedding"]
        assert stored.shape[0] == store.dimension


class TestVectorStorePersistence:
    def test_disk_persistence_creates_files(self, tmp_path: Path) -> None:
        index_path = tmp_path / "vector_index"
        store = VectorStore(
            dimension=384,
            storage_type=StorageType.DISK,
            model_name=TEST_MODEL,
            index_path=index_path,
            collection_name="test_collection",
        )
        store.store_word(_sample_entry())

        persisted = list(index_path.rglob("*.sqlite*"))
        assert persisted, "Expected vector store persistence files to be created"
