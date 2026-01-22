from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_directed_chordless_cycle_undirected(self):
    g = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (5, 1), (0, 2)])
    expected_cycles = [(0, 2, 3, 4, 5), (1, 2, 3, 4, 5)]
    self.check_cycle_algorithm(g, expected_cycles, chordless=True)
    g = nx.DiGraph()
    nx.add_cycle(g, range(5))
    nx.add_cycle(g, range(4, 9))
    g.add_edge(7, 3)
    expected_cycles = [(0, 1, 2, 3, 4), (3, 4, 5, 6, 7), (4, 5, 6, 7, 8)]
    self.check_cycle_algorithm(g, expected_cycles, chordless=True)
    g.add_edge(3, 7)
    expected_cycles = [(0, 1, 2, 3, 4), (3, 7), (4, 5, 6, 7, 8)]
    self.check_cycle_algorithm(g, expected_cycles, chordless=True)
    expected_cycles = [(3, 7)]
    self.check_cycle_algorithm(g, expected_cycles, chordless=True, length_bound=4)
    g.remove_edge(7, 3)
    expected_cycles = [(0, 1, 2, 3, 4), (4, 5, 6, 7, 8)]
    self.check_cycle_algorithm(g, expected_cycles, chordless=True)
    g = nx.DiGraph(((i, j) for i in range(10) for j in range(i)))
    expected_cycles = []
    self.check_cycle_algorithm(g, expected_cycles, chordless=True)