import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_in_edges_dir(self):
    G = self.P3
    assert sorted(G.in_edges()) == [(0, 1), (1, 2)]
    assert sorted(G.in_edges(0)) == []
    assert sorted(G.in_edges(2)) == [(1, 2)]