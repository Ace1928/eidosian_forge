import pytest
import networkx as nx
from networkx.utils import pairwise
def test_zero_cycle(self):
    G = nx.cycle_graph(5, create_using=nx.DiGraph())
    G.add_edge(2, 3, weight=-4)
    nx.goldberg_radzik(G, 1)
    nx.bellman_ford_predecessor_and_distance(G, 1)
    G.add_edge(2, 3, weight=-4.0001)
    pytest.raises(nx.NetworkXUnbounded, nx.bellman_ford_predecessor_and_distance, G, 1)
    pytest.raises(nx.NetworkXUnbounded, nx.goldberg_radzik, G, 1)