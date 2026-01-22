from functools import partial
import pytest
import networkx as nx
def test_bfs_edges_reverse(self):
    D = nx.DiGraph()
    D.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)])
    edges = nx.bfs_edges(D, source=4, reverse=True)
    assert list(edges) == [(4, 2), (4, 3), (2, 1), (1, 0)]