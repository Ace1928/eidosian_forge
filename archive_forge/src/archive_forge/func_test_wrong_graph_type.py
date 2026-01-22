import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_wrong_graph_type(self):
    G = nx.DiGraph()
    G_edges = [[0, 1], [0, 2], [0, 3]]
    G.add_edges_from(G_edges)
    pytest.raises(nx.NetworkXNotImplemented, nx.inverse_line_graph, G)
    G = nx.MultiGraph()
    G_edges = [[0, 1], [0, 2], [0, 3]]
    G.add_edges_from(G_edges)
    pytest.raises(nx.NetworkXNotImplemented, nx.inverse_line_graph, G)