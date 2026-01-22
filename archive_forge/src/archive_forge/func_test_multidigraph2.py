import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_multidigraph2(self):
    G = nx.MultiDiGraph([(0, 1), (0, 1), (0, 1), (1, 2)])
    L = nx.line_graph(G)
    assert edges_equal(L.edges(), [((0, 1, 0), (1, 2, 0)), ((0, 1, 1), (1, 2, 0)), ((0, 1, 2), (1, 2, 0))])