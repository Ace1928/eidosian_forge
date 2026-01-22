import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_graph_chain(self):
    G = self.Graph([(0, 1), (1, 2)])
    DG = G.to_directed(as_view=True)
    SDG = DG.subgraph([0, 1])
    RSDG = SDG.reverse(copy=False)
    assert G is DG._graph
    assert DG is SDG._graph
    assert SDG is RSDG._graph