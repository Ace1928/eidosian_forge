"""Tests for GraphManager functionality.

This module tests the GraphManager using the real networkx library.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from word_forge.graph.graph_manager import GraphManager
from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager


def test_build_graph_from_db(tmp_path):
    db = DBManager(db_path=tmp_path / "test.db")
    db.insert_or_update_word("alpha", "first")
    db.insert_or_update_word("beta", "second")
    db.insert_relationship("alpha", "beta", "synonym")

    manager = GraphManager(db_manager=db)
    manager.build_graph()

    terms = {data["term"] for _, data in manager.g.nodes(data=True)}
    assert {"alpha", "beta"} <= terms
    assert manager.g.number_of_edges() == 1


def test_graph_includes_emotional_relationships(tmp_path):
    db = DBManager(db_path=tmp_path / "emotions.db")
    db.insert_or_update_word("alpha", "first")

    emotion_manager = EmotionManager(db_manager=db)
    alpha_id = db.get_word_id("alpha")
    emotion_manager.set_word_emotion(alpha_id, 0.8, 0.7)

    manager = GraphManager(db_manager=db)
    manager.build_graph()

    joy_id = db.get_word_id("joy")
    edge_data = manager.g.get_edge_data(alpha_id, joy_id)

    assert edge_data
    assert edge_data.get("dimension") == "emotional"
    assert abs(edge_data.get("valence") - 0.8) < 1e-6
    assert abs(edge_data.get("arousal") - 0.7) < 1e-6
