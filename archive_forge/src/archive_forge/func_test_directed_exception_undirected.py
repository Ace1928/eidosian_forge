import pytest
import networkx as nx
def test_directed_exception_undirected():
    graph = nx.Graph([(0, 1), (2, 3)])
    with pytest.raises(nx.NetworkXNotImplemented):
        G = nx.directed_edge_swap(graph)