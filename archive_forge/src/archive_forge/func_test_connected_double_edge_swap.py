import pytest
import networkx as nx
def test_connected_double_edge_swap():
    graph = nx.barabasi_albert_graph(200, 1)
    degrees = sorted((d for n, d in graph.degree()))
    G = nx.connected_double_edge_swap(graph, 40, seed=1)
    assert nx.is_connected(graph)
    assert degrees == sorted((d for n, d in graph.degree()))