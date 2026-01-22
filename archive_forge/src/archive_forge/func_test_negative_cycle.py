import pytest
import networkx as nx
from networkx.utils import pairwise
def test_negative_cycle(self):
    G = nx.DiGraph()
    G.add_weighted_edges_from([('0', '3', 3), ('0', '1', -5), ('1', '0', -5), ('0', '2', 2), ('1', '2', 4), ('2', '3', 1)])
    pytest.raises(nx.NetworkXUnbounded, nx.johnson, G)
    G = nx.Graph()
    G.add_weighted_edges_from([('0', '3', 3), ('0', '1', -5), ('1', '0', -5), ('0', '2', 2), ('1', '2', 4), ('2', '3', 1)])
    pytest.raises(nx.NetworkXUnbounded, nx.johnson, G)