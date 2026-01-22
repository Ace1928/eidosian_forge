from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
def test_number_of_edges_selfloops(self):
    G = self.K3
    G.add_edge(0, 0)
    G.add_edge(0, 0)
    G.add_edge(0, 0, key='parallel edge')
    G.remove_edge(0, 0, key='parallel edge')
    assert G.number_of_edges(0, 0) == 2
    G.remove_edge(0, 0)
    assert G.number_of_edges(0, 0) == 1