import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_symmetry_with_custom_matching(self):
    print('G2 is edge (a,b) and G3 is edge (a,a)')
    print('but node order for G2 is (a,b) while for G3 it is (b,a)')
    a, b = ('A', 'B')
    G2 = nx.Graph()
    G2.add_nodes_from((a, b))
    G2.add_edges_from([(a, b)])
    G3 = nx.Graph()
    G3.add_nodes_from((b, a))
    G3.add_edges_from([(a, a)])
    for G in (G2, G3):
        for n in G:
            G.nodes[n]['attr'] = n
        for e in G.edges:
            G.edges[e]['attr'] = e
    match = lambda x, y: x == y
    print('Starting G2 to G3 GED calculation')
    assert nx.graph_edit_distance(G2, G3, node_match=match, edge_match=match) == 1
    print('Starting G3 to G2 GED calculation')
    assert nx.graph_edit_distance(G3, G2, node_match=match, edge_match=match) == 1