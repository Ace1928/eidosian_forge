import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_reverse_hashable(self):

    class Foo:
        pass
    x = Foo()
    y = Foo()
    G = nx.DiGraph()
    G.add_edge(x, y)
    assert nodes_equal(G.nodes(), G.reverse().nodes())
    assert [(y, x)] == list(G.reverse().edges())