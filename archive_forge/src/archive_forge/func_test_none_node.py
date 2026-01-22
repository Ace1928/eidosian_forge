import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_none_node(self):
    G = self.Graph()
    with pytest.raises(ValueError):
        G.add_node(None)
    with pytest.raises(ValueError):
        G.add_nodes_from([None])
    with pytest.raises(ValueError):
        G.add_edge(0, None)
    with pytest.raises(ValueError):
        G.add_edges_from([(0, None)])