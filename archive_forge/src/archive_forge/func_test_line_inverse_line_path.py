import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_line_inverse_line_path(self):
    G = nx.path_graph(10)
    H = nx.line_graph(G)
    J = nx.inverse_line_graph(H)
    assert nx.is_isomorphic(G, J)