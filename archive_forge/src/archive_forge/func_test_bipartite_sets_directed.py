import pytest
import networkx as nx
from networkx.algorithms import bipartite
def test_bipartite_sets_directed(self):
    G = nx.path_graph(4)
    D = G.to_directed()
    X, Y = bipartite.sets(D)
    assert X == {0, 2}
    assert Y == {1, 3}