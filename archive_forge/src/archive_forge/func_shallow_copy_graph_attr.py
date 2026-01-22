import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def shallow_copy_graph_attr(self, H, G):
    assert G.graph['foo'] == H.graph['foo']
    G.graph['foo'].append(1)
    assert G.graph['foo'] == H.graph['foo']