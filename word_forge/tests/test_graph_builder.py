"""Tests for word_forge.graph.graph_builder using real data."""

from __future__ import annotations

import pytest

from word_forge.graph.graph_builder import GraphBuilder, GraphUpdateMetrics


class TestGraphUpdateMetrics:
    def test_default_values(self) -> None:
        metrics = GraphUpdateMetrics()
        assert metrics.new_nodes == 0
        assert metrics.new_edges == 0
        assert metrics.processed_words == 0
        assert metrics.max_last_refreshed == 0.0
        assert metrics.full_rebuild is False

    def test_custom_values(self) -> None:
        metrics = GraphUpdateMetrics(
            new_nodes=2,
            new_edges=3,
            processed_words=4,
            max_last_refreshed=1.0,
            full_rebuild=True,
        )
        assert metrics.new_nodes == 2
        assert metrics.new_edges == 3
        assert metrics.processed_words == 4
        assert metrics.max_last_refreshed == 1.0
        assert metrics.full_rebuild is True

    def test_immutability(self) -> None:
        metrics = GraphUpdateMetrics()
        with pytest.raises(AttributeError):
            metrics.new_nodes = 1  # type: ignore


class TestGraphBuilderIntegration:
    def test_build_graph_populates_nodes(self, graph_manager) -> None:
        builder = GraphBuilder(graph_manager)
        builder.build_graph()
        assert graph_manager.get_node_count() > 0

    def test_update_graph_detects_new_nodes(
        self, graph_manager, populated_db_manager
    ) -> None:
        builder = GraphBuilder(graph_manager)

        populated_db_manager.insert_or_update_word(
            "serenity",
            "state of calm",
            "noun",
        )

        new_nodes = builder.update_graph()
        assert new_nodes >= 1

    def test_verify_database_tables(self, graph_manager) -> None:
        builder = GraphBuilder(graph_manager)
        assert builder.verify_database_tables() is True
