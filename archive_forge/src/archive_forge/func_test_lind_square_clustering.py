import pytest
import networkx as nx
def test_lind_square_clustering(self):
    """Test C4 for figure 1 Lind et al (2005)"""
    G = nx.Graph([(1, 2), (1, 3), (1, 6), (1, 7), (2, 4), (2, 5), (3, 4), (3, 5), (6, 7), (7, 8), (6, 8), (7, 9), (7, 10), (6, 11), (6, 12), (2, 13), (2, 14), (3, 15), (3, 16)])
    G1 = G.subgraph([1, 2, 3, 4, 5, 13, 14, 15, 16])
    G2 = G.subgraph([1, 6, 7, 8, 9, 10, 11, 12])
    assert nx.square_clustering(G, [1])[1] == 3 / 43
    assert nx.square_clustering(G1, [1])[1] == 2 / 6
    assert nx.square_clustering(G2, [1])[1] == 1 / 5