import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_out_edges_dir(self):
    G = self.P3
    assert sorted(G.out_edges()) == [(0, 1), (1, 2)]
    assert sorted(G.out_edges(0)) == [(0, 1)]
    assert sorted(G.out_edges(2)) == []