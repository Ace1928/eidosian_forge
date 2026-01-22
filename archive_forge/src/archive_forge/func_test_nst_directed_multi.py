import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nst_directed_multi(self):
    G = nx.MultiDiGraph()
    G = nx.cycle_graph(3, G)
    G.add_edge(1, 2)
    assert np.isclose(nx.number_of_spanning_trees(G, root=0), 2)