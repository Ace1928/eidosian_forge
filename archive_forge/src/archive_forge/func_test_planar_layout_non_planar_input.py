import pytest
import networkx as nx
def test_planar_layout_non_planar_input(self):
    G = nx.complete_graph(9)
    pytest.raises(nx.NetworkXException, nx.planar_layout, G)