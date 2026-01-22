from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def four_cycles(g, **kwargs):
    yield from (c for c in nx.simple_cycles(g, **kwargs) if len(c) == 4)