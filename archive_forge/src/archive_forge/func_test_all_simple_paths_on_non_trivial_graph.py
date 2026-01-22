import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_all_simple_paths_on_non_trivial_graph():
    """you may need to draw this graph to make sure it is reasonable"""
    G = nx.path_graph(5, create_using=nx.DiGraph())
    G.add_edges_from([(0, 5), (1, 5), (1, 3), (5, 4), (4, 2), (4, 3)])
    paths = nx.all_simple_paths(G, 1, [2, 3])
    assert {tuple(p) for p in paths} == {(1, 2), (1, 3, 4, 2), (1, 5, 4, 2), (1, 3), (1, 2, 3), (1, 5, 4, 3), (1, 5, 4, 2, 3)}
    paths = nx.all_simple_paths(G, 1, [2, 3], cutoff=3)
    assert {tuple(p) for p in paths} == {(1, 2), (1, 3, 4, 2), (1, 5, 4, 2), (1, 3), (1, 2, 3), (1, 5, 4, 3)}
    paths = nx.all_simple_paths(G, 1, [2, 3], cutoff=2)
    assert {tuple(p) for p in paths} == {(1, 2), (1, 3), (1, 2, 3)}