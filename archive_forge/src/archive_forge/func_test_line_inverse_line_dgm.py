import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_line_inverse_line_dgm(self):
    G = nx.dorogovtsev_goltsev_mendes_graph(4)
    H = nx.line_graph(G)
    J = nx.inverse_line_graph(H)
    assert nx.is_isomorphic(G, J)