from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_simple_cycles_bound_error(self):
    with pytest.raises(ValueError):
        G = nx.DiGraph()
        for c in nx.simple_cycles(G, -1):
            assert False
    with pytest.raises(ValueError):
        G = nx.Graph()
        for c in nx.simple_cycles(G, -1):
            assert False
    with pytest.raises(ValueError):
        G = nx.Graph()
        for c in nx.chordless_cycles(G, -1):
            assert False
    with pytest.raises(ValueError):
        G = nx.DiGraph()
        for c in nx.chordless_cycles(G, -1):
            assert False