import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_remove_nodes_from(self):
    G = self.K3.copy()
    G.remove_nodes_from([0, 1])
    assert G.adj == {2: {}}
    G.remove_nodes_from([-1])