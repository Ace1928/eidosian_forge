from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_tree_graph(self):
    tg = nx.balanced_tree(3, 3)
    assert not nx.minimum_cycle_basis(tg)