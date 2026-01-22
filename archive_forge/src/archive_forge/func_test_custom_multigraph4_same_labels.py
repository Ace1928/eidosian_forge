import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_multigraph4_same_labels(self):
    G1 = nx.MultiGraph()
    edges1 = [(1, 2), (1, 2), (2, 2), (2, 3), (3, 8), (3, 8), (3, 4), (4, 5), (4, 5), (4, 5), (4, 6), (3, 6), (3, 6), (6, 6), (8, 7), (7, 7), (8, 9), (9, 9), (8, 9), (8, 9), (5, 9), (10, 11), (11, 12), (12, 13), (11, 13), (10, 10), (10, 11), (11, 13)]
    mapped = {1: 'n', 2: 'm', 3: 'l', 4: 'j', 5: 'k', 6: 'i', 7: 'g', 8: 'h', 9: 'f', 10: 'b', 11: 'a', 12: 'd', 13: 'e'}
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G1.add_edges_from([(2, 2), (2, 3), (2, 8), (3, 4)])
    G2.add_edges_from([('m', 'm'), ('m', 'l'), ('m', 'h'), ('l', 'j')])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    H1 = nx.MultiGraph(G1.subgraph([2, 3, 4, 6, 10, 11, 12, 13]))
    H2 = nx.MultiGraph(G2.subgraph([mapped[2], mapped[3], mapped[8], mapped[9], mapped[10], mapped[11], mapped[12], mapped[13]]))
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert not m
    H2.remove_edges_from([(mapped[3], mapped[2]), (mapped[9], mapped[8]), (mapped[2], mapped[2])])
    H2.add_edges_from([(mapped[9], mapped[9]), (mapped[2], mapped[8])])
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m
    H1.remove_node(12)
    H2.remove_node(mapped[12])
    H1.add_edge(13, 13)
    H2.add_edge(mapped[13], mapped[13])
    H1.add_edges_from([(3, 13), (6, 11)])
    H2.add_edges_from([(mapped[8], mapped[10]), (mapped[2], mapped[11])])
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m
    H1.remove_edges_from([(2, 2), (3, 6)])
    H1.add_edges_from([(6, 6), (2, 3)])
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert not m