import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_add_edge(self):
    G = self.Graph()
    G.add_edge(0, 1)
    assert G.adj == {0: {1: {}}, 1: {}}
    assert G.succ == {0: {1: {}}, 1: {}}
    assert G.pred == {0: {}, 1: {0: {}}}
    G = self.Graph()
    G.add_edge(*(0, 1))
    assert G.adj == {0: {1: {}}, 1: {}}
    assert G.succ == {0: {1: {}}, 1: {}}
    assert G.pred == {0: {}, 1: {0: {}}}
    with pytest.raises(ValueError, match='None cannot be a node'):
        G.add_edge(None, 3)