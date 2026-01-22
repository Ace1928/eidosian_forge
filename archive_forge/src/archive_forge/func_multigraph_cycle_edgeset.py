from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def multigraph_cycle_edgeset(c):
    if len(c) <= 2:
        return frozenset(cycle_edges(c))
    else:
        return frozenset(map(frozenset, cycle_edges(c)))