import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
from networkx import convert_node_labels_to_integers as cnlti
from networkx.classes.tests import dispatch_interface
def test_number_connected_components2(self):
    ncc = nx.number_connected_components
    assert ncc(self.grid) == 1