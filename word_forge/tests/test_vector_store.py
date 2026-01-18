"""Integration tests for word_forge.vectorizer.vector_store."""

from __future__ import annotations

from pathlib import Path

import pytest

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
