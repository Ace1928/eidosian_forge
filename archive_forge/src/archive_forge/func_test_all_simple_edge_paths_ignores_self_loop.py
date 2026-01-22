import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_all_simple_edge_paths_ignores_self_loop():
    G = nx.Graph([(0, 0), (0, 1), (1, 1), (1, 2)])
    assert list(nx.all_simple_edge_paths(G, 0, 2)) == [[(0, 1), (1, 2)]]