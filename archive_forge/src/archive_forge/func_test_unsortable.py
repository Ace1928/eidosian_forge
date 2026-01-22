from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_unsortable(self):
    G = nx.DiGraph()
    nx.add_cycle(G, ['a', 1])
    c = list(nx.simple_cycles(G))
    assert len(c) == 1