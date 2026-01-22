import itertools
import networkx as nx
def test_visibility_graph__flat_series__path_graph():
    series = [0] * 10
    expected_node_count = len(series)
    actual_graph = nx.visibility_graph(series)
    assert actual_graph.number_of_nodes() == expected_node_count
    assert actual_graph.number_of_edges() == expected_node_count - 1
    assert nx.is_isomorphic(actual_graph, nx.path_graph(expected_node_count))