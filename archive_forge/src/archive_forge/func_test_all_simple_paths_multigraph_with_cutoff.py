import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_all_simple_paths_multigraph_with_cutoff():
    G = nx.MultiGraph([(1, 2), (1, 2), (1, 10), (10, 2)])
    paths = list(nx.all_simple_paths(G, 1, 2, cutoff=1))
    assert len(paths) == 2
    assert {tuple(p) for p in paths} == {(1, 2), (1, 2)}