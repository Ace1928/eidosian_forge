from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
@staticmethod
def edgeset_function(g):
    if g.is_directed():
        return directed_cycle_edgeset
    elif g.is_multigraph():
        return multigraph_cycle_edgeset
    else:
        return undirected_cycle_edgeset