import math
import pytest
import networkx as nx
from networkx.generators.interval_graph import interval_graph
from networkx.utils import edges_equal
def test_interval_graph_3(self):
    intervals = [(1, 4), [3, 5], [2.5, 4]]
    expected_graph = nx.Graph()
    expected_graph.add_nodes_from([(1, 4), (3, 5), (2.5, 4)])
    e1 = ((1, 4), (3, 5))
    e2 = ((1, 4), (2.5, 4))
    e3 = ((3, 5), (2.5, 4))
    expected_graph.add_edges_from([e1, e2, e3])
    actual_g = interval_graph(intervals)
    assert set(actual_g.nodes) == set(expected_graph.nodes)
    assert edges_equal(expected_graph, actual_g)