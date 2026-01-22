from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def is_cyclic_permutation(self, a, b):
    n = len(a)
    if len(b) != n:
        return False
    l = a + a
    return any((l[i:i + n] == b for i in range(n)))