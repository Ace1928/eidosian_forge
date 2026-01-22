import pytest
import networkx as nx
from networkx.utils import pairwise
def test_dijkstra_predecessor1(self):
    G = nx.path_graph(4)
    assert nx.dijkstra_predecessor_and_distance(G, 0) == ({0: [], 1: [0], 2: [1], 3: [2]}, {0: 0, 1: 1, 2: 2, 3: 3})