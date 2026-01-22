from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_chordless_cycles_graph(self):
    G = nx.Graph()
    nx.add_cycle(G, range(5))
    nx.add_cycle(G, range(4, 12))
    expected = [[*range(5)], [*range(4, 12)]]
    self.check_cycle_algorithm(G, expected, chordless=True)
    self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 5], length_bound=5, chordless=True)
    G.add_edge(7, 3)
    expected.append([*range(3, 8)])
    expected.append([4, 3, 7, 8, 9, 10, 11])
    self.check_cycle_algorithm(G, expected, chordless=True)
    self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 5], length_bound=5, chordless=True)