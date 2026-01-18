"""Tests for graph analysis functionality with emotional context filtering.

This module tests the GraphManager's emotional subgraph extraction and
context integration features using the real networkx library.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from word_forge.graph.graph_manager import GraphManager
from word_forge.database.database_manager import DBManager


def create_manager(tmp_path):
    db = DBManager(tmp_path / "test.db")
    return GraphManager(db_manager=db)


def test_emotional_subgraph_context_filtering(tmp_path):
    manager = create_manager(tmp_path)
    a = manager.add_word_node("happy", {"valence": 0.8})
    b = manager.add_word_node("joyful", {"valence": 0.9})
    c = manager.add_word_node("sad", {"valence": -0.6})

    manager.add_relationship(a, b, "joy_associated", dimension="emotional", weight=1.0)
    manager.add_relationship(
        a, c, "sadness_associated", dimension="emotional", weight=1.0
    )

    manager.integrate_emotional_context(
        "clinical", {"sadness_associated": 0.2, "joy_associated": 1.0}
    )

    sub = manager.get_emotional_subgraph(
        "happy", depth=1, context="clinical", min_intensity=0.5
    )

    terms = {manager.query.get_term_by_id(n) for n in sub.nodes}
    assert "sad" not in terms
    assert "joyful" in terms
