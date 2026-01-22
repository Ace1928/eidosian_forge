import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nst_directed_cycle_graph(self):
    G = nx.DiGraph()
    G = nx.cycle_graph(7, G)
    assert np.isclose(nx.number_of_spanning_trees(G, root=0), 1)