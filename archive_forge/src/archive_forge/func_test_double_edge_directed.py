import pytest
import networkx as nx
def test_double_edge_directed():
    graph = nx.DiGraph([(0, 1), (2, 3)])
    with pytest.raises(nx.NetworkXError, match='not defined for directed graphs.'):
        G = nx.double_edge_swap(graph)