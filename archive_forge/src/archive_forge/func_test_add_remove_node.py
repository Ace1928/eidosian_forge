import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_add_remove_node(self):
    G = self.G()
    G.add_node('A')
    assert G.has_node('A')
    G.remove_node('A')
    assert not G.has_node('A')