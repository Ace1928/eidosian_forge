from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_chordless_cycles_clique(self):
    g_family = [self.K(n) for n in range(2, 15)]
    expected = [0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364]
    self.check_cycle_enumeration_integer_sequence(g_family, expected, chordless=True)
    expected = [(n * n - n) // 2 for n in range(15)]
    g_family = [self.D(n) for n in range(15)]
    self.check_cycle_enumeration_integer_sequence(g_family, expected, chordless=True)