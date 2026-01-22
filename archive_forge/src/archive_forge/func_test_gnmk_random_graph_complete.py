import numbers
import pytest
import networkx as nx
from ..generators import (
def test_gnmk_random_graph_complete(self):
    n = 10
    m = 20
    edges = 200
    G = gnmk_random_graph(n, m, edges)
    assert len(G) == n + m
    assert nx.is_bipartite(G)
    X, Y = nx.algorithms.bipartite.sets(G)
    assert set(range(n)) == X
    assert set(range(n, n + m)) == Y
    assert edges == len(list(G.edges()))