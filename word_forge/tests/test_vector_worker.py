"""Integration tests for word_forge.vectorizer.vector_worker."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from word_forge.configs.config_essentials import StorageType
from word_forge.database.database_manager import DBManager
from word_forge.vectorizer.vector_store import VectorStore
from word_forge.vectorizer.vector_worker import EmbeddingError, VectorWorker


TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _wait_until(predicate, timeout: float = 30.0, interval: float = 0.25) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_vector_worker_processes_updates(tmp_path: Path) -> None:
    db = DBManager(db_path=tmp_path / "vectors.db")
    db.insert_or_update_word("alpha", "first")
    db.insert_or_update_word("beta", "second")

    store = VectorStore(
        dimension=384,
        storage_type=StorageType.MEMORY,
        demo_mode=True,
        model_name=TEST_MODEL,
    )
    worker = VectorWorker(db, store, TEST_MODEL)

    words = worker._get_all_words()
    worker._process_words(words)
    worker.last_processed = time.time()

    assert store.collection.count() > 0

    time.sleep(0.02)
    db.insert_or_update_word("gamma", "third")
    db.insert_or_update_word("alpha", "updated")

    updated_words = worker._get_all_words()
    updated_terms = {word.term for word in updated_words}
    assert updated_terms == {"alpha", "gamma"}

    worker._process_words(updated_words)
    assert store.collection.count() >= 3


def test_vector_worker_thread_updates_state(tmp_path: Path) -> None:
    db = DBManager(db_path=tmp_path / "threaded_vectors.db")
    db.insert_or_update_word("delta", "fourth")

    store = VectorStore(
        dimension=384,
        storage_type=StorageType.MEMORY,
        demo_mode=True,
        model_name=TEST_MODEL,
    )
    worker = VectorWorker(db, store, TEST_MODEL, poll_interval=0.2, daemon=False)

    worker.start()
    try:
        assert _wait_until(lambda: worker.last_processed is not None, timeout=30.0)
    finally:
        worker.stop()
        worker.join(timeout=5)


def test_invalid_model_name_raises_embedding_error(tmp_path: Path) -> None:
    db = DBManager(db_path=tmp_path / "invalid_model.db")
    store = VectorStore(
        dimension=384,
        storage_type=StorageType.MEMORY,
        demo_mode=True,
        model_name=TEST_MODEL,
    )

    with pytest.raises(EmbeddingError):
        VectorWorker(db, store, "invalid/does-not-exist")
