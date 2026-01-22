import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_all_simple_edge_paths_directed():
    G = nx.DiGraph()
    nx.add_path(G, [1, 2, 3])
    nx.add_path(G, [3, 2, 1])
    paths = nx.all_simple_edge_paths(G, 1, 3)
    assert {tuple(p) for p in paths} == {((1, 2), (2, 3))}