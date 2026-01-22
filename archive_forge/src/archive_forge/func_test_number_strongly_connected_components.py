import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_number_strongly_connected_components(self):
    ncc = nx.number_strongly_connected_components
    for G, C in self.gc:
        assert ncc(G) == len(C)