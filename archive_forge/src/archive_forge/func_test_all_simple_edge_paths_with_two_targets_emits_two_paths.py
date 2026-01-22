import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_all_simple_edge_paths_with_two_targets_emits_two_paths():
    G = nx.path_graph(4)
    G.add_edge(2, 4)
    paths = nx.all_simple_edge_paths(G, 0, [3, 4])
    assert {tuple(p) for p in paths} == {((0, 1), (1, 2), (2, 3)), ((0, 1), (1, 2), (2, 4))}