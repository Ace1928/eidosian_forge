"""Integration tests for word_forge.graph.graph_query."""

from __future__ import annotations

import pytest

from word_forge.exceptions import NodeNotFoundError


def test_query_counts(graph_manager) -> None:
    query = graph_manager.query
    assert query.get_node_count() > 0
    assert query.get_edge_count() > 0


def test_get_related_terms(graph_manager) -> None:
    related = graph_manager.query.get_related_terms("happiness")
    assert "joy" in {term.lower() for term in related}


def test_get_term_by_id(graph_manager) -> None:
    node_id = graph_manager.query.get_node_id("happiness")
    assert node_id is not None
    assert graph_manager.query.get_term_by_id(node_id) == "happiness"


def test_get_graph_info(graph_manager) -> None:
    info = graph_manager.query.get_graph_info()
    assert info["nodes"] > 0
    assert info["edges"] > 0
    assert info["sample_nodes"]


def test_get_subgraph_invalid_term(graph_manager) -> None:
    with pytest.raises(NodeNotFoundError):
        graph_manager.query.get_subgraph("nonexistent-term")
