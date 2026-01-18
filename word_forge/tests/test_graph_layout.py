"""Tests for graph layout functionality.

This module tests the GraphManager's layout system using the real networkx library.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from word_forge.graph.graph_manager import GraphManager
from word_forge.database.database_manager import DBManager


def create_manager(tmp_path):
    db = DBManager(tmp_path / "layout.db")
    return GraphManager(db_manager=db)


def test_incremental_layout_updates_existing_positions(tmp_path):
    manager = create_manager(tmp_path)
    a = manager.add_word_node("a")
    b = manager.add_word_node("b")

    manager.layout.compute_layout()
    pos_before = {
        a: manager._positions[a],
        b: manager._positions[b],
    }

    c = manager.add_word_node("c")

    assert np.allclose(manager._positions[a], pos_before[a])
    assert np.allclose(manager._positions[b], pos_before[b])
    assert c in manager._positions
