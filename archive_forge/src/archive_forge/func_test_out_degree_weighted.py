import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_out_degree_weighted(self):
    G = self.K3.copy()
    G.add_edge(0, 1, weight=0.3, other=1.2)
    assert sorted(G.out_degree(weight='weight')) == [(0, 1.3), (1, 2), (2, 2)]
    assert dict(G.out_degree(weight='weight')) == {0: 1.3, 1: 2, 2: 2}
    assert G.out_degree(0, weight='weight') == 1.3
    assert sorted(G.out_degree(weight='other')) == [(0, 2.2), (1, 2), (2, 2)]
    assert dict(G.out_degree(weight='other')) == {0: 2.2, 1: 2, 2: 2}
    assert G.out_degree(0, weight='other') == 2.2
    assert list(G.out_degree(iter([0]), weight='other')) == [(0, 2.2)]