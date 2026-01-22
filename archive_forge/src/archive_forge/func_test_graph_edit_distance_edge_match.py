import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_graph_edit_distance_edge_match(self):
    G1 = path_graph(6)
    G2 = path_graph(6)
    for e, attr in G1.edges.items():
        attr['color'] = 'red' if min(e) % 2 == 0 else 'blue'
    for e, attr in G2.edges.items():
        attr['color'] = 'red' if min(e) // 3 == 0 else 'blue'
    assert graph_edit_distance(G1, G2) == 0
    assert graph_edit_distance(G1, G2, edge_match=lambda e1, e2: e1['color'] == e2['color']) == 2