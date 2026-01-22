import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_digraph1(self):
    G = nx.DiGraph([(0, 1), (0, 2), (0, 3)])
    L = nx.line_graph(G)
    assert L.adj == {(0, 1): {}, (0, 2): {}, (0, 3): {}}