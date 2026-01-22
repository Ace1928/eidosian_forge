import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_bipartite_color(self):
    G = nx.path_graph(4)
    c = bipartite.color(G)
    assert c == {0: 1, 1: 0, 2: 1, 3: 0}