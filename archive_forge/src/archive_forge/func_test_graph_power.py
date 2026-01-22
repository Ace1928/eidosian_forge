import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_graph_power():
    G = nx.cycle_graph(7)
    G.add_edge(6, 7)
    G.add_edge(7, 8)
    G.add_edge(8, 9)
    G.add_edge(9, 2)
    H = nx.power(G, 2)
    assert edges_equal(list(H.edges()), [(0, 1), (0, 2), (0, 5), (0, 6), (0, 7), (1, 9), (1, 2), (1, 3), (1, 6), (2, 3), (2, 4), (2, 8), (2, 9), (3, 4), (3, 5), (3, 9), (4, 5), (4, 6), (5, 6), (5, 7), (6, 7), (6, 8), (7, 8), (7, 9), (8, 9)])