import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_nbunch_iter(self):
    G = self.K3
    assert nodes_equal(G.nbunch_iter(), self.k3nodes)
    assert nodes_equal(G.nbunch_iter(0), [0])
    assert nodes_equal(G.nbunch_iter([0, 1]), [0, 1])
    assert nodes_equal(G.nbunch_iter([-1]), [])
    assert nodes_equal(G.nbunch_iter('foo'), [])
    bunch = G.nbunch_iter(-1)
    with pytest.raises(nx.NetworkXError, match='is not a node or a sequence'):
        list(bunch)
    bunch = G.nbunch_iter([0, 1, 2, {}])
    with pytest.raises(nx.NetworkXError, match='in sequence nbunch is not a valid node'):
        list(bunch)