import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nst_disconnected(self):
    G = nx.empty_graph(2)
    assert np.isclose(nx.number_of_spanning_trees(G), 0)