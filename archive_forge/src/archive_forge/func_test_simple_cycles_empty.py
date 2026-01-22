from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_simple_cycles_empty(self):
    G = nx.DiGraph()
    assert list(nx.simple_cycles(G)) == []