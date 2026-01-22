import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal
def test_biadjacency_matrix_weight(self):
    G = nx.path_graph(5)
    G.add_edge(0, 1, weight=2, other=4)
    X = [1, 3]
    Y = [0, 2, 4]
    M = bipartite.biadjacency_matrix(G, X, weight='weight')
    assert M[0, 0] == 2
    M = bipartite.biadjacency_matrix(G, X, weight='other')
    assert M[0, 0] == 4