import pytest
import networkx as nx
from networkx.utils import pairwise
def test_find_negative_cycle_longer_cycle(self):
    G = nx.cycle_graph(5, create_using=nx.DiGraph())
    nx.add_cycle(G, [3, 5, 6, 7, 8, 9])
    G.add_edge(1, 2, weight=-30)
    assert nx.find_negative_cycle(G, 1) == [0, 1, 2, 3, 4, 0]
    assert nx.find_negative_cycle(G, 7) == [2, 3, 4, 0, 1, 2]