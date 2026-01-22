import itertools
import networkx as nx
def test_visibility_graph__convex_series__complete_graph():
    series = [i ** 2 for i in range(10)]
    expected_series_length = len(series)
    actual_graph = nx.visibility_graph(series)
    assert actual_graph.number_of_nodes() == expected_series_length
    assert actual_graph.number_of_edges() == 45
    assert nx.is_isomorphic(actual_graph, nx.complete_graph(expected_series_length))