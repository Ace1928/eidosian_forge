import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def shallow_copy_attrdict(self, H, G):
    self.shallow_copy_graph_attr(H, G)
    self.shallow_copy_node_attr(H, G)
    self.shallow_copy_edge_attr(H, G)