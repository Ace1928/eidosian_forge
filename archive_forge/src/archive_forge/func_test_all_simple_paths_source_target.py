import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_all_simple_paths_source_target():
    G = nx.path_graph(4)
    paths = nx.all_simple_paths(G, 1, 1)
    assert list(paths) == []