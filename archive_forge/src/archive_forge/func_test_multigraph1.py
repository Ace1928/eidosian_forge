import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_multigraph1(self):
    G = nx.MultiGraph([(0, 1), (0, 1), (1, 0), (0, 2), (2, 0), (0, 3)])
    L = nx.line_graph(G)
    assert edges_equal(L.edges(), [((0, 3, 0), (0, 1, 0)), ((0, 3, 0), (0, 2, 0)), ((0, 3, 0), (0, 2, 1)), ((0, 3, 0), (0, 1, 1)), ((0, 3, 0), (0, 1, 2)), ((0, 1, 0), (0, 1, 1)), ((0, 1, 0), (0, 2, 0)), ((0, 1, 0), (0, 1, 2)), ((0, 1, 0), (0, 2, 1)), ((0, 1, 1), (0, 1, 2)), ((0, 1, 1), (0, 2, 0)), ((0, 1, 1), (0, 2, 1)), ((0, 1, 2), (0, 2, 0)), ((0, 1, 2), (0, 2, 1)), ((0, 2, 0), (0, 2, 1))])