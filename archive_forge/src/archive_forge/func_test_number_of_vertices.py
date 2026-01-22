from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_number_of_vertices(self):
    m, n = (5, 6)
    G = nx.grid_2d_graph(m, n)
    assert len(G) == m * n