import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
def test_to_undirected_reciprocal(self):
    G = self.Graph()
    G.add_edge(1, 2)
    assert G.to_undirected().has_edge(1, 2)
    assert not G.to_undirected(reciprocal=True).has_edge(1, 2)
    G.add_edge(2, 1)
    assert G.to_undirected(reciprocal=True).has_edge(1, 2)