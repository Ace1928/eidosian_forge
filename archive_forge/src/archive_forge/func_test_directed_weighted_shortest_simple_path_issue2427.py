import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_directed_weighted_shortest_simple_path_issue2427():
    G = nx.DiGraph()
    G.add_edge('IN', 'OUT', weight=2)
    G.add_edge('IN', 'A', weight=1)
    G.add_edge('IN', 'B', weight=2)
    G.add_edge('B', 'OUT', weight=2)
    assert list(nx.shortest_simple_paths(G, 'IN', 'OUT', weight='weight')) == [['IN', 'OUT'], ['IN', 'B', 'OUT']]
    G = nx.DiGraph()
    G.add_edge('IN', 'OUT', weight=10)
    G.add_edge('IN', 'A', weight=1)
    G.add_edge('IN', 'B', weight=1)
    G.add_edge('B', 'OUT', weight=1)
    assert list(nx.shortest_simple_paths(G, 'IN', 'OUT', weight='weight')) == [['IN', 'B', 'OUT'], ['IN', 'OUT']]