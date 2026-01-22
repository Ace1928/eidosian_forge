import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nst_complete_graph(self):
    N = 5
    G = nx.complete_graph(N)
    assert np.isclose(nx.number_of_spanning_trees(G), N ** (N - 2))