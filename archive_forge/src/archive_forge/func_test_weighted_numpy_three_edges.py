import pytest
import networkx as nx
def test_weighted_numpy_three_edges():
    XG3 = nx.Graph()
    XG3.add_weighted_edges_from([[0, 1, 2], [1, 2, 12], [2, 3, 1], [3, 4, 5], [4, 5, 1], [5, 0, 10]])
    dist = nx.floyd_warshall_numpy(XG3)
    assert dist[0, 3] == 15