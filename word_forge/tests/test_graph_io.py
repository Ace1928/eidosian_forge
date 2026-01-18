"""Integration tests for word_forge.graph.graph_io."""

from __future__ import annotations

from pathlib import Path

import pytest

from word_forge.database.database_manager import DBManager
from word_forge.graph.graph_manager import GraphManager


def test_save_and_load_gexf(graph_manager, tmp_path: Path) -> None:
    output_path = tmp_path / "graph_export.gexf"
    graph_manager.io.save_to_gexf(str(output_path))

    assert output_path.exists()

    new_db = DBManager(db_path=tmp_path / "graph_load.db")
    new_manager = GraphManager(db_manager=new_db)
    new_manager.io.load_from_gexf(str(output_path))

    assert new_manager.get_node_count() > 0


def test_save_empty_graph_no_crash(tmp_path: Path) -> None:
    db = DBManager(db_path=tmp_path / "empty_graph.db")
    manager = GraphManager(db_manager=db)

    # Empty graph should not raise
    manager.io.save_to_gexf(str(tmp_path / "empty.gexf"))
