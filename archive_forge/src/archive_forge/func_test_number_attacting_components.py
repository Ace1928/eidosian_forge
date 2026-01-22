import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_number_attacting_components(self):
    assert nx.number_attracting_components(self.G1) == 3
    assert nx.number_attracting_components(self.G2) == 1
    assert nx.number_attracting_components(self.G3) == 2
    assert nx.number_attracting_components(self.G4) == 0