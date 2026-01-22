import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_multigraph5_same_labels(self):
    G1 = nx.MultiGraph()
    edges1 = [(1, 5), (1, 2), (1, 4), (2, 3), (2, 6), (3, 4), (3, 7), (4, 8), (5, 8), (5, 6), (6, 7), (7, 8)]
    mapped = {1: 'a', 2: 'h', 3: 'd', 4: 'i', 5: 'g', 6: 'b', 7: 'j', 8: 'c'}
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G1.add_edges_from([(1, 2), (1, 2), (3, 7), (8, 8), (8, 8), (7, 8), (2, 3), (5, 6)])
    G2.add_edges_from([('a', 'h'), ('a', 'h'), ('d', 'j'), ('c', 'c'), ('c', 'c'), ('j', 'c'), ('d', 'h'), ('g', 'b')])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G2.remove_edges_from([('a', 'h'), ('a', 'h'), ('d', 'j'), ('c', 'c'), ('c', 'c'), ('j', 'c'), ('d', 'h'), ('g', 'b')])
    G2.add_edges_from([('d', 'i'), ('a', 'h'), ('g', 'b'), ('g', 'b'), ('i', 'i'), ('i', 'i'), ('b', 'j'), ('d', 'j')])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m