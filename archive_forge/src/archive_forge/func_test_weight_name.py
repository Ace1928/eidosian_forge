import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_weight_name():
    G = nx.cycle_graph(7)
    nx.set_edge_attributes(G, 1, 'weight')
    nx.set_edge_attributes(G, 1, 'foo')
    G.adj[1][2]['foo'] = 7
    paths = list(nx.shortest_simple_paths(G, 0, 3, weight='foo'))
    solution = [[0, 6, 5, 4, 3], [0, 1, 2, 3]]
    assert paths == solution