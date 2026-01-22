import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_create_multigraph():
    """Tests that using a MultiGraph creates multiple edges."""
    G = nx.path_graph(3, create_using=nx.MultiGraph())
    G.add_edge(0, 1)
    G.add_edge(0, 0)
    G.add_edge(0, 2)
    actual = nx.contracted_nodes(G, 0, 2)
    expected = nx.MultiGraph()
    expected.add_edge(0, 1)
    expected.add_edge(0, 1)
    expected.add_edge(0, 1)
    expected.add_edge(0, 0)
    expected.add_edge(0, 0)
    assert edges_equal(actual.edges, expected.edges)