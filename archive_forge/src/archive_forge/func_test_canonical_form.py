from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_canonical_form(self):
    T = nx.Graph()
    T.add_edges_from([(0, 1), (0, 2), (0, 3)])
    T.add_edges_from([(1, 4), (1, 5)])
    T.add_edges_from([(3, 6), (3, 7)])
    root = 0
    actual = nx.to_nested_tuple(T, root, canonical_form=True)
    expected = ((), ((), ()), ((), ()))
    assert actual == expected