import math
from functools import partial
import pytest
import networkx as nx
def test_notimplemented(self):
    G = nx.DiGraph([(0, 1), (1, 2)])
    G.add_nodes_from([0, 1, 2], community=0)
    assert pytest.raises(nx.NetworkXNotImplemented, self.func, G, [(0, 2)])
    G = nx.MultiGraph([(0, 1), (1, 2)])
    G.add_nodes_from([0, 1, 2], community=0)
    assert pytest.raises(nx.NetworkXNotImplemented, self.func, G, [(0, 2)])
    G = nx.MultiDiGraph([(0, 1), (1, 2)])
    G.add_nodes_from([0, 1, 2], community=0)
    assert pytest.raises(nx.NetworkXNotImplemented, self.func, G, [(0, 2)])