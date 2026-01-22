import pytest
import networkx as nx
def test_nodelist():
    G = nx.path_graph(7)
    dist = nx.floyd_warshall_numpy(G, nodelist=[3, 5, 4, 6, 2, 1, 0])
    assert dist[0, 3] == 3
    assert dist[0, 1] == 2
    assert dist[6, 2] == 4
    pytest.raises(nx.NetworkXError, nx.floyd_warshall_numpy, G, [1, 3])
    pytest.raises(nx.NetworkXError, nx.floyd_warshall_numpy, G, list(range(9)))