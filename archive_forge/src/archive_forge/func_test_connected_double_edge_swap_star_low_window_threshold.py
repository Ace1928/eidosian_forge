import pytest
import networkx as nx
def test_connected_double_edge_swap_star_low_window_threshold():
    graph = nx.star_graph(40)
    degrees = sorted((d for n, d in graph.degree()))
    G = nx.connected_double_edge_swap(graph, 1, _window_threshold=0, seed=4)
    assert nx.is_connected(graph)
    assert degrees == sorted((d for n, d in graph.degree()))