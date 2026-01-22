import pytest
import networkx as nx
def test_induced_nodes(self):
    G = nx.generators.classic.path_graph(10)
    Induced_nodes = nx.find_induced_nodes(G, 1, 9, 2)
    assert Induced_nodes == {1, 2, 3, 4, 5, 6, 7, 8, 9}
    pytest.raises(nx.NetworkXTreewidthBoundExceeded, nx.find_induced_nodes, G, 1, 9, 1)
    Induced_nodes = nx.find_induced_nodes(self.chordal_G, 1, 6)
    assert Induced_nodes == {1, 2, 4, 6}
    pytest.raises(nx.NetworkXError, nx.find_induced_nodes, self.non_chordal_G, 1, 5)