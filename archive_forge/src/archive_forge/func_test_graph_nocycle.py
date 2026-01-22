from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_graph_nocycle(self):
    G = nx.Graph(self.edges)
    pytest.raises(nx.exception.NetworkXNoCycle, nx.find_cycle, G, self.nodes)