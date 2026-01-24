"""Integration tests for ConversationManager using real dependencies."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Skip all tests in this module if vector dependencies are unavailable
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

pytestmark = pytest.mark.skipif(
    not _VECTOR_AVAILABLE,
    reason="Vector dependencies (chromadb, sentence-transformers) not installed",
)

from word_forge.conversation.conversation_manager import ConversationManager
from word_forge.conversation.conversation_models import (
    AffectiveLexicalLanguageModel,
    EidosianIdentityModel,
    LightweightLanguageModel,
    ReflexiveLanguageModel,
)
from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.parser.language_model import ModelState
from word_forge.queue.queue_manager import QueueManager
from word_forge.vectorizer.vector_store import VectorStore
from word_forge.configs.config_essentials import StorageType

TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "sshleifer/tiny-gpt2"


def _create_manager(tmp_path: Path) -> ConversationManager:
    db = DBManager(db_path=tmp_path / "conv.db")
    emotion_manager = EmotionManager(db)
    graph_manager = GraphManager(db_manager=db)
    vector_store = VectorStore(
        dimension=384,
        storage_type=StorageType.MEMORY,
        demo_mode=True,
        model_name=TEST_MODEL,
        db_manager=db,
    )

    llm_state = ModelState(model_name=LLM_MODEL)
    identity_model = EidosianIdentityModel(llm_state=llm_state)

    return ConversationManager(
        db_manager=db,
        emotion_manager=emotion_manager,
        graph_manager=graph_manager,
        vector_store=vector_store,
        reflexive_model=ReflexiveLanguageModel(llm_state=llm_state),
        lightweight_model=LightweightLanguageModel(llm_state=llm_state),
        affective_model=AffectiveLexicalLanguageModel(llm_state=llm_state),
        identity_model=identity_model,
        queue_manager=QueueManager[str](),
    )


def test_conversation_flow(tmp_path: Path) -> None:
    cm = _create_manager(tmp_path)
    conv_id = cm.start_conversation().unwrap()

    add_res = cm.add_message(conv_id, "User", "Hello", generate_response=False)
    assert add_res.is_success

    conv = cm.get_conversation(conv_id).unwrap()
    assert len(conv["messages"]) == 1
    assert conv["messages"][0]["text"] == "Hello"

    end_res = cm.end_conversation(conv_id)
    assert end_res.is_success


def test_end_nonexistent_conversation(tmp_path: Path) -> None:
    cm = _create_manager(tmp_path)
    res = cm.end_conversation(999)
    assert res.is_failure
    assert res.error and res.error.code == "CONVERSATION_NOT_FOUND"


def test_add_message_empty(tmp_path: Path) -> None:
    cm = _create_manager(tmp_path)
    conv_id = cm.start_conversation().unwrap()

    with pytest.raises(ValueError):
        cm.add_message(conv_id, "User", " ")


def test_generate_response_pipeline(tmp_path: Path) -> None:
    cm = _create_manager(tmp_path)
    conv_id = cm.start_conversation().unwrap()

    result = cm.add_message(conv_id, "User", "What is joy?", generate_response=True)
    assert result.is_success

    conv = cm.get_conversation(conv_id).unwrap()
    assert len(conv["messages"]) >= 2
