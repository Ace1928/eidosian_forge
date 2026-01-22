from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_no_cycle(self):
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 0), (3, 1), (3, 2)])
    pytest.raises(nx.NetworkXNoCycle, nx.find_cycle, G, source=0)
    pytest.raises(nx.NetworkXNoCycle, nx.find_cycle, G)