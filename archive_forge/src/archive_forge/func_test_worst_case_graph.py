from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_worst_case_graph(self):
    for k in range(3, 10):
        G = self.worst_case_graph(k)
        l = len(list(nx.simple_cycles(G)))
        assert l == 3 * k