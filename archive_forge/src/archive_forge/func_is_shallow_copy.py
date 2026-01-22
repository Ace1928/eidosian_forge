import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def is_shallow_copy(self, H, G):
    self.graphs_equal(H, G)
    self.shallow_copy_attrdict(H, G)