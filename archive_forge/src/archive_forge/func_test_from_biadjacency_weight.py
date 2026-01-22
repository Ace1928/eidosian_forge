import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal
def test_from_biadjacency_weight(self):
    M = sparse.csc_matrix([[1, 2], [0, 3]])
    B = bipartite.from_biadjacency_matrix(M)
    assert edges_equal(B.edges(), [(0, 2), (0, 3), (1, 3)])
    B = bipartite.from_biadjacency_matrix(M, edge_attribute='weight')
    e = [(0, 2, {'weight': 1}), (0, 3, {'weight': 2}), (1, 3, {'weight': 3})]
    assert edges_equal(B.edges(data=True), e)