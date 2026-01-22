from functools import partial
import pytest
import networkx as nx
def test_bfs_edges(self):
    edges = nx.bfs_edges(self.G, source=0)
    assert list(edges) == [(0, 1), (1, 2), (1, 3), (2, 4)]