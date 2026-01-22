import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_intersection_all_different_node_sets():
    G = nx.Graph()
    H = nx.Graph()
    R = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 6, 7])
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(6, 7)
    H.add_nodes_from([1, 2, 3, 4])
    H.add_edge(2, 3)
    H.add_edge(3, 4)
    R.add_nodes_from([1, 2, 3, 4, 8, 9])
    R.add_edge(2, 3)
    R.add_edge(4, 1)
    R.add_edge(8, 9)
    I = nx.intersection_all([G, H, R])
    assert set(I.nodes()) == {1, 2, 3, 4}
    assert sorted(I.edges()) == [(2, 3)]