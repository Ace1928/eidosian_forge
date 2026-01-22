import itertools
import networkx as nx
def test_visibility_graph_cyclic_series():
    series = list(itertools.islice(itertools.cycle((2, 1, 3)), 17))
    expected_node_count = len(series)
    actual_graph = nx.visibility_graph(series)
    assert actual_graph.number_of_nodes() == expected_node_count
    assert actual_graph.number_of_edges() == 25