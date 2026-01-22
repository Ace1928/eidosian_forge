from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
def different_attrdict(self, H, G):
    old_foo = H[1][2][0]['foo']
    H.adj[1][2][0]['foo'] = 'baz'
    assert G._adj != H._adj
    H.adj[1][2][0]['foo'] = old_foo
    assert G._adj == H._adj
    old_foo = H.nodes[0]['foo']
    H.nodes[0]['foo'] = 'baz'
    assert G._node != H._node
    H.nodes[0]['foo'] = old_foo
    assert G._node == H._node