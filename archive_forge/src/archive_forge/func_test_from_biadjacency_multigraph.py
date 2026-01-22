import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal
def test_from_biadjacency_multigraph(self):
    M = sparse.csc_matrix([[1, 2], [0, 3]])
    B = bipartite.from_biadjacency_matrix(M, create_using=nx.MultiGraph())
    assert edges_equal(B.edges(), [(0, 2), (0, 3), (0, 3), (1, 3), (1, 3), (1, 3)])