import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_multigraph2_different_labels(self):
    G1 = nx.MultiGraph()
    mapped = {1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'G', 7: 'B', 6: 'F'}
    edges1 = [(1, 2), (1, 2), (1, 5), (1, 5), (1, 5), (5, 6), (2, 3), (2, 3), (2, 4), (3, 4), (3, 4), (4, 5), (4, 5), (4, 5), (2, 7), (2, 7), (2, 7)]
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
    nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    assert m == mapped
    G1.remove_edge(2, 7)
    G1.add_edge(5, 6)
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    G2.remove_edge('B', 'C')
    G2.add_edge('G', 'F')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    assert m == mapped
    G1.remove_node(3)
    G2.remove_node('D')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G1.remove_edge(1, 2)
    G1.remove_edge(2, 7)
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    G2.add_edges_from([('A', 'C'), ('C', 'E'), ('C', 'E')])
    G2.remove_edges_from([('A', 'G'), ('A', 'G'), ('F', 'G'), ('E', 'G'), ('E', 'G')])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    for n1, n2 in zip(G1.nodes(), G2.nodes()):
        G1.nodes[n1]['label'] = 'blue'
        G2.nodes[n2]['label'] = 'blue'
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m