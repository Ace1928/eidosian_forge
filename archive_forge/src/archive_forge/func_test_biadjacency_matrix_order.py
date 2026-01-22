import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal
def test_biadjacency_matrix_order(self):
    G = nx.path_graph(5)
    G.add_edge(0, 1, weight=2)
    X = [3, 1]
    Y = [4, 2, 0]
    M = bipartite.biadjacency_matrix(G, X, Y, weight='weight')
    assert M[1, 2] == 2