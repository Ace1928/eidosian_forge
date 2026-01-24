"""CLI smoke tests without stubs or mocks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from word_forge.utils.nltk_utils import ensure_nltk_data

# Check if vector dependencies are available
def _package_available(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def _vector_available() -> bool:
    if (
        importlib.util.find_spec("chromadb") is None
        or importlib.util.find_spec("sentence_transformers") is None
    ):
        return False
    try:
        import word_forge.vectorizer.vector_store as vector_store
    except Exception:
        return False
    return vector_store.SentenceTransformer is not None


_VECTOR_AVAILABLE = _vector_available()

TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "sshleifer/tiny-gpt2"


def test_cli_version_command() -> None:
    from word_forge import forge

    assert forge.main(["--version"]) == 0


def test_cli_setup_nltk_command() -> None:
    from word_forge import forge

    ensure_nltk_data()
    assert forge.main(["setup-nltk"]) == 0


@pytest.mark.skipif(
    not _VECTOR_AVAILABLE,
    reason="Vector dependencies (chromadb, sentence-transformers) not installed",
)
def test_cli_start_command(tmp_path: Path) -> None:
    from word_forge import forge

    ensure_nltk_data()

    result = forge.main(
        [
            "start",
            "happy",
            "--minutes",
            "0.01",
            "--workers",
            "1",
            "--db-path",
            str(tmp_path / "cli_start.db"),
            "--vector-model",
            TEST_MODEL,
            "--llm-model",
            LLM_MODEL,
        ]
    )
    assert result == 0


def test_cli_graph_build_command() -> None:
    from word_forge import forge

    assert (
        forge.main(["graph", "build", "--timeout", "10", "--poll-interval", "0.5"]) == 0
    )


@pytest.mark.skipif(
    not _VECTOR_AVAILABLE,
    reason="Vector dependencies (chromadb, sentence-transformers) not installed",
)
def test_cli_vector_index_command(tmp_path: Path) -> None:
    from word_forge import forge

    result = forge.main(
        [
            "vector",
            "index",
            "--embedder",
            TEST_MODEL,
            "--timeout",
            "20",
            "--poll-interval",
            "0.5",
        ]
    )
    assert result == 0
