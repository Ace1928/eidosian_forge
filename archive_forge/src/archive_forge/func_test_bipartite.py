import random
import pytest
import networkx as nx
def test_bipartite():
    G = nx.complete_bipartite_graph(12, 34)
    indep = nx.maximal_independent_set(G, [4, 5, 9, 10])
    assert sorted(indep) == list(range(12))