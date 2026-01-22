import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_nbunch_is_set(self):
    G = self.G()
    nbunch = set('ABCDEFGHIJKL')
    G.add_nodes_from(nbunch)
    assert G.has_node('L')