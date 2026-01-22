import pytest
import networkx as nx
from networkx import NetworkXNotImplemented
def test_is_strongly_connected(self):
    for G, C in self.gc:
        if len(C) == 1:
            assert nx.is_strongly_connected(G)
        else:
            assert not nx.is_strongly_connected(G)