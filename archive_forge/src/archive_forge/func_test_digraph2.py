import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_digraph2(self):
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    L = nx.line_graph(G)
    assert edges_equal(L.edges(), [((0, 1), (1, 2)), ((1, 2), (2, 3))])