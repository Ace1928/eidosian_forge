import pytest
import networkx as nx
def test_gaussian_random_partition_graph():
    G = nx.gaussian_random_partition_graph(100, 10, 10, 0.3, 0.01)
    assert len(G) == 100
    G = nx.gaussian_random_partition_graph(100, 10, 10, 0.3, 0.01, directed=True)
    assert len(G) == 100
    G = nx.gaussian_random_partition_graph(100, 10, 10, 0.3, 0.01, directed=False, seed=42)
    assert len(G) == 100
    assert not isinstance(G, nx.DiGraph)
    G = nx.gaussian_random_partition_graph(100, 10, 10, 0.3, 0.01, directed=True, seed=42)
    assert len(G) == 100
    assert isinstance(G, nx.DiGraph)
    pytest.raises(nx.NetworkXError, nx.gaussian_random_partition_graph, 100, 101, 10, 1, 0)
    G = nx.gaussian_random_partition_graph(10, 0.5, 0.5, 0.5, 0.5, seed=1)
    assert len(G) == 10