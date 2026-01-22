import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_add_edges_from(self):
    G = self.Graph()
    G.add_edges_from([(0, 1), (0, 2, {'data': 3})], data=2)
    assert G.adj == {0: {1: {'data': 2}, 2: {'data': 3}}, 1: {}, 2: {}}
    assert G.succ == {0: {1: {'data': 2}, 2: {'data': 3}}, 1: {}, 2: {}}
    assert G.pred == {0: {}, 1: {0: {'data': 2}}, 2: {0: {'data': 3}}}
    with pytest.raises(nx.NetworkXError):
        G.add_edges_from([(0,)])
    with pytest.raises(nx.NetworkXError):
        G.add_edges_from([(0, 1, 2, 3)])
    with pytest.raises(TypeError):
        G.add_edges_from([0])
    with pytest.raises(ValueError, match='None cannot be a node'):
        G.add_edges_from([(None, 3), (3, 2)])