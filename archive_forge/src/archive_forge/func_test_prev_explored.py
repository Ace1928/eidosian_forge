from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_prev_explored(self):
    G = nx.DiGraph()
    G.add_edges_from([(1, 0), (2, 0), (1, 2), (2, 1)])
    pytest.raises(nx.NetworkXNoCycle, nx.find_cycle, G, source=0)
    x = list(nx.find_cycle(G, 1))
    x_ = [(1, 2), (2, 1)]
    assert x == x_
    x = list(nx.find_cycle(G, 2))
    x_ = [(2, 1), (1, 2)]
    assert x == x_
    x = list(nx.find_cycle(G))
    x_ = [(1, 2), (2, 1)]
    assert x == x_