import itertools
import pytest
import networkx as nx
def test_is_coloring(self):
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    coloring = {0: 0, 1: 1, 2: 0}
    assert is_coloring(G, coloring)
    coloring[0] = 1
    assert not is_coloring(G, coloring)
    assert not is_equitable(G, coloring)