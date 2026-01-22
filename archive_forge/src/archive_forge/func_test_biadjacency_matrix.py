import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal
def test_biadjacency_matrix(self):
    tops = [2, 5, 10]
    bots = [5, 10, 15]
    for i in range(len(tops)):
        G = bipartite.random_graph(tops[i], bots[i], 0.2)
        top = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
        M = bipartite.biadjacency_matrix(G, top)
        assert M.shape[0] == tops[i]
        assert M.shape[1] == bots[i]