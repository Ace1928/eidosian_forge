import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_without_self_loops():
    """Tests for node contraction without preserving -loops."""
    G = nx.cycle_graph(4)
    actual = nx.contracted_nodes(G, 0, 1, self_loops=False)
    expected = nx.complete_graph(3)
    assert nx.is_isomorphic(actual, expected)