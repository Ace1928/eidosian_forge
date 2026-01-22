import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_balanced_tree_star(self):
    t = nx.balanced_tree(r=2, h=1)
    assert is_isomorphic(t, nx.star_graph(2))
    t = nx.balanced_tree(r=5, h=1)
    assert is_isomorphic(t, nx.star_graph(5))
    t = nx.balanced_tree(r=10, h=1)
    assert is_isomorphic(t, nx.star_graph(10))