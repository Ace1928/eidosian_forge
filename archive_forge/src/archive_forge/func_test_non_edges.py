import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_non_edges(self):
    graph = nx.complete_graph(5)
    nedges = list(nx.non_edges(graph))
    assert len(nedges) == 0
    graph = nx.path_graph(4)
    expected = [(0, 2), (0, 3), (1, 3)]
    nedges = list(nx.non_edges(graph))
    for u, v in expected:
        assert (u, v) in nedges or (v, u) in nedges
    graph = nx.star_graph(4)
    expected = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    nedges = list(nx.non_edges(graph))
    for u, v in expected:
        assert (u, v) in nedges or (v, u) in nedges
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 2), (2, 0), (2, 1)])
    expected = [(0, 1), (1, 0), (1, 2)]
    nedges = list(nx.non_edges(graph))
    for e in expected:
        assert e in nedges