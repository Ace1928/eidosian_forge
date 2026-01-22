import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_directed_node_contraction():
    """Tests for node contraction in a directed graph."""
    G = nx.DiGraph(nx.cycle_graph(4))
    actual = nx.contracted_nodes(G, 0, 1)
    expected = nx.DiGraph(nx.cycle_graph(3))
    expected.add_edge(0, 0)
    expected.add_edge(0, 0)
    assert nx.is_isomorphic(actual, expected)