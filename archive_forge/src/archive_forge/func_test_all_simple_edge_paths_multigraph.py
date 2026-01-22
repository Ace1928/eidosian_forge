import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_all_simple_edge_paths_multigraph():
    G = nx.MultiGraph([(1, 2), (1, 2)])
    paths = nx.all_simple_edge_paths(G, 1, 1)
    assert list(paths) == []
    nx.add_path(G, [3, 1, 10, 2])
    paths = list(nx.all_simple_edge_paths(G, 1, 2))
    assert len(paths) == 3
    assert {tuple(p) for p in paths} == {((1, 2, 0),), ((1, 2, 1),), ((1, 10, 0), (10, 2, 0))}