from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
def deepcopy_edge_attr(self, H, G):
    assert G[1][2][0]['foo'] == H[1][2][0]['foo']
    G[1][2][0]['foo'].append(1)
    assert G[1][2][0]['foo'] != H[1][2][0]['foo']