import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_undirected_node_contraction_no_copy():
    """Tests for node contraction in an undirected graph
    by making changes in place."""
    G = nx.cycle_graph(4)
    actual = nx.contracted_nodes(G, 0, 1, copy=False)
    expected = nx.cycle_graph(3)
    expected.add_edge(0, 0)
    assert nx.is_isomorphic(actual, G)
    assert nx.is_isomorphic(actual, expected)