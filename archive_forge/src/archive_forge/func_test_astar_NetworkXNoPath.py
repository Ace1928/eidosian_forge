import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_NetworkXNoPath(self):
    """Tests that exception is raised when there exists no
        path between source and target"""
    G = nx.gnp_random_graph(10, 0.2, seed=10)
    with pytest.raises(nx.NetworkXNoPath):
        nx.astar_path(G, 4, 9)