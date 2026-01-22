import pytest
import networkx as nx
from networkx.utils import pairwise
def test_zero_cycle_smoke(self):
    D = nx.DiGraph()
    D.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 1, -2)])
    nx.bellman_ford_path(D, 1, 3)
    nx.dijkstra_path(D, 1, 3)
    nx.bidirectional_dijkstra(D, 1, 3)