from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def check_cycle_enumeration_integer_sequence(self, g_family, cycle_counts, length_bound=None, chordless=False, algorithm=None):
    for g, num_cycles in zip(g_family, cycle_counts):
        self.check_cycle_algorithm(g, num_cycles, length_bound=length_bound, chordless=chordless, algorithm=algorithm)