import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_nbunch_iter_node_format_raise(self):
    G = self.Graph()
    nbunch = [('x', set())]
    with pytest.raises(nx.NetworkXError):
        list(G.nbunch_iter(nbunch))