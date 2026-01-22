from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)