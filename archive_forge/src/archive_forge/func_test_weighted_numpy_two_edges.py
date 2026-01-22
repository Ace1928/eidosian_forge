import pytest
import networkx as nx
def test_weighted_numpy_two_edges():
    XG4 = nx.Graph()
    XG4.add_weighted_edges_from([[0, 1, 2], [1, 2, 2], [2, 3, 1], [3, 4, 1], [4, 5, 1], [5, 6, 1], [6, 7, 1], [7, 0, 1]])
    dist = nx.floyd_warshall_numpy(XG4)
    assert dist[0, 2] == 4