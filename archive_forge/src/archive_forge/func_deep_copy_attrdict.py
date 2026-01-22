import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def deep_copy_attrdict(self, H, G):
    self.deepcopy_graph_attr(H, G)
    self.deepcopy_node_attr(H, G)
    self.deepcopy_edge_attr(H, G)