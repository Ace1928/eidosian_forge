import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_multigraph3_different_labels(self):
    G1 = nx.MultiGraph()
    mapped = {1: 9, 2: 8, 3: 7, 4: 6, 5: 3, 8: 5, 9: 4, 7: 1, 6: 2}
    edges1 = [(1, 2), (1, 3), (1, 3), (2, 3), (2, 3), (3, 4), (4, 5), (4, 7), (4, 9), (4, 9), (4, 9), (5, 8), (5, 8), (8, 9), (8, 9), (5, 6), (6, 7), (6, 7), (6, 7), (5, 2)]
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
    nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    assert m == mapped
    G1.remove_edge(4, 9)
    G2.remove_edge(4, 6)
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    assert m == mapped
    G1.add_edges_from([(4, 9), (1, 2), (1, 2)])
    G1.remove_edges_from([(1, 3), (1, 3)])
    G2.add_edges_from([(3, 5), (7, 9)])
    G2.remove_edge(8, 9)
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    for n1, n2 in zip(G1.nodes(), G2.nodes()):
        G1.nodes[n1]['label'] = 'blue'
        G2.nodes[n2]['label'] = 'blue'
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G1.add_node(10)
    G2.add_node('Z')
    G1.nodes[10]['label'] = 'green'
    G2.nodes['Z']['label'] = 'green'
    G1.add_edges_from([(10, 10), (10, 10)])
    G2.add_edges_from([('Z', 'Z')])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    G1.remove_edge(10, 10)
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G1.add_edges_from([(10, 3), (10, 4)])
    G2.add_edges_from([('Z', 8), ('Z', 3)])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G1.remove_node(4)
    G2.remove_node(3)
    G1.add_edges_from([(5, 6), (5, 6), (5, 7)])
    G2.add_edges_from([(1, 6), (1, 6), (6, 2)])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m