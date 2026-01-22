import math
import pytest
import networkx as nx
from networkx.generators.interval_graph import interval_graph
from networkx.utils import edges_equal
def test_interval_graph_2(self):
    intervals = [(1, 2), [3, 5], [6, 8], (9, 10)]
    expected_graph = nx.Graph()
    expected_graph.add_nodes_from([(1, 2), (3, 5), (6, 8), (9, 10)])
    actual_g = interval_graph(intervals)
    assert set(actual_g.nodes) == set(expected_graph.nodes)
    assert edges_equal(expected_graph, actual_g)