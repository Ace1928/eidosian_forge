import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_line_inverse_line_star(self):
    G = nx.star_graph(20)
    H = nx.line_graph(G)
    J = nx.inverse_line_graph(H)
    assert nx.is_isomorphic(G, J)