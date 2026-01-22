import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nst_multigraph(self):
    G = nx.MultiGraph()
    G.add_edge(1, 2)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 3)
    assert np.isclose(nx.number_of_spanning_trees(G), 5)