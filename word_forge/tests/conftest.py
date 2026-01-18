"""Shared pytest fixtures for Word Forge test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def db_manager(tmp_path: Path):
    """Create an isolated DBManager instance for testing."""
    from word_forge.database.database_manager import DBManager

    db_path = tmp_path / "test.db"
    manager = DBManager(db_path=db_path)
    manager.create_tables()
    return manager


@pytest.fixture
def populated_db_manager(db_manager):
    """Create a DBManager with sample data."""
    db_manager.insert_or_update_word("happiness", "a state of joy", "noun")
    db_manager.insert_or_update_word("sadness", "a state of sorrow", "noun")
    db_manager.insert_or_update_word("joy", "intense happiness", "noun")
    db_manager.insert_or_update_word("sorrow", "deep sadness", "noun")
    db_manager.insert_or_update_word("anger", "strong displeasure", "noun")
    db_manager.insert_or_update_word("love", "deep affection", "noun")
    db_manager.insert_or_update_word("fear", "emotional response to danger", "noun")
    db_manager.insert_or_update_word("surprise", "unexpected emotion", "noun")

    db_manager.insert_relationship("happiness", "joy", "synonym")
    db_manager.insert_relationship("happiness", "sadness", "antonym")
    db_manager.insert_relationship("sadness", "sorrow", "synonym")
    db_manager.insert_relationship("joy", "happiness", "synonym")
    db_manager.insert_relationship("love", "happiness", "evokes")

    return db_manager


@pytest.fixture
def emotion_manager(db_manager):
    from word_forge.emotion.emotion_manager import EmotionManager

    return EmotionManager(db_manager)


@pytest.fixture
def graph_manager(populated_db_manager):
    from word_forge.graph.graph_manager import GraphManager

    manager = GraphManager(db_manager=populated_db_manager)
    manager.build_graph()
    return manager
