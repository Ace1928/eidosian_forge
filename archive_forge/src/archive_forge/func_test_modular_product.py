import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_modular_product():
    G = nx.path_graph(3)
    H = nx.path_graph(4)
    M = nx.modular_product(G, H)
    assert len(M) == len(G) * len(H)
    assert edges_equal(list(M.edges()), [((0, 0), (1, 1)), ((0, 0), (2, 2)), ((0, 0), (2, 3)), ((0, 1), (1, 0)), ((0, 1), (1, 2)), ((0, 1), (2, 3)), ((0, 2), (1, 1)), ((0, 2), (1, 3)), ((0, 2), (2, 0)), ((0, 3), (1, 2)), ((0, 3), (2, 0)), ((0, 3), (2, 1)), ((1, 0), (2, 1)), ((1, 1), (2, 0)), ((1, 1), (2, 2)), ((1, 2), (2, 1)), ((1, 2), (2, 3)), ((1, 3), (2, 2))])