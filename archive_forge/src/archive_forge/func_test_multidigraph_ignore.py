from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_multidigraph_ignore(self):
    G = nx.MultiDiGraph(self.edges)
    x = list(nx.find_cycle(G, self.nodes, orientation='ignore'))
    x_ = [(0, 1, 0, FORWARD), (1, 0, 0, FORWARD)]
    assert x[0] == x_[0]
    assert x[1][:2] == x_[1][:2]
    assert x[1][3] == x_[1][3]