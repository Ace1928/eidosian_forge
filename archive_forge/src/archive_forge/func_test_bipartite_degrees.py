import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_bipartite_degrees(self):
    G = nx.path_graph(5)
    X = {1, 3}
    Y = {0, 2, 4}
    u, d = bipartite.degrees(G, Y)
    assert dict(u) == {1: 2, 3: 2}
    assert dict(d) == {0: 1, 2: 2, 4: 1}