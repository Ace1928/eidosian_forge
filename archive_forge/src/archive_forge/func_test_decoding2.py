from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_decoding2(self):
    sequence = [2, 4, 0, 1, 3, 3]
    tree = nx.from_prufer_sequence(sequence)
    assert nodes_equal(list(tree), list(range(8)))
    edges = [(0, 1), (0, 4), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]
    assert edges_equal(list(tree.edges()), edges)