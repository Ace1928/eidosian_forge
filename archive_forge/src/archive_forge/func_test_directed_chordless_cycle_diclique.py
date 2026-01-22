from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_directed_chordless_cycle_diclique(self):
    g_family = [self.D(n) for n in range(10)]
    expected_cycles = [(n * n - n) // 2 for n in range(10)]
    self.check_cycle_enumeration_integer_sequence(g_family, expected_cycles, chordless=True)
    expected_cycles = [(n * n - n) // 2 for n in range(10)]
    self.check_cycle_enumeration_integer_sequence(g_family, expected_cycles, length_bound=2)