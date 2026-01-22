import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_create2(self):
    G = nx.Graph([(0, 1), (1, 2), (2, 3)])
    L = nx.line_graph(G, create_using=nx.DiGraph())
    assert edges_equal(L.edges(), [((0, 1), (1, 2)), ((1, 2), (2, 3))])