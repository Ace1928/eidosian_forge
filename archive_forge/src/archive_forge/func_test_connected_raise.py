import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
from networkx import convert_node_labels_to_integers as cnlti
from networkx.classes.tests import dispatch_interface
def test_connected_raise(self):
    with pytest.raises(NetworkXNotImplemented):
        next(nx.connected_components(self.DG))
    pytest.raises(NetworkXNotImplemented, nx.number_connected_components, self.DG)
    pytest.raises(NetworkXNotImplemented, nx.node_connected_component, self.DG, 1)
    pytest.raises(NetworkXNotImplemented, nx.is_connected, self.DG)
    pytest.raises(nx.NetworkXPointlessConcept, nx.is_connected, nx.Graph())