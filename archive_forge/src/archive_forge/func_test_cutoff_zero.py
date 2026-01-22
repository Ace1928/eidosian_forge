import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_cutoff_zero():
    G = nx.complete_graph(4)
    paths = nx.all_simple_paths(G, 0, 3, cutoff=0)
    assert [list(p) for p in paths] == []
    paths = nx.all_simple_paths(nx.MultiGraph(G), 0, 3, cutoff=0)
    assert [list(p) for p in paths] == []