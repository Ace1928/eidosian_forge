import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nst_weight(self):
    G = nx.Graph()
    G.add_edge(1, 2, weight=1)
    G.add_edge(1, 3, weight=1)
    G.add_edge(2, 3, weight=2)
    assert np.isclose(nx.number_of_spanning_trees(G), 3)
    assert np.isclose(nx.number_of_spanning_trees(G, weight='weight'), 5)