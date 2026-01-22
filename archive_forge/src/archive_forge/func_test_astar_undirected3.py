import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_undirected3(self):
    XG4 = nx.Graph()
    edges = [(0, 1, 2), (1, 2, 2), (2, 3, 1), (3, 4, 1), (4, 5, 1), (5, 6, 1), (6, 7, 1), (7, 0, 1)]
    XG4.add_weighted_edges_from(edges)
    assert nx.astar_path(XG4, 0, 2) == [0, 1, 2]
    assert nx.astar_path_length(XG4, 0, 2) == 4