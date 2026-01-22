import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_multigraph1_different_labels(self):
    G1 = nx.MultiGraph()
    mapped = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'Z', 6: 'E'}
    edges1 = [(1, 2), (1, 3), (1, 4), (1, 4), (1, 4), (2, 3), (2, 6), (2, 6), (3, 4), (3, 4), (5, 1), (5, 1), (5, 2), (5, 2)]
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
    nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    assert m == mapped
    G1.remove_edge(1, 4)
    G1.add_edge(1, 5)
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    G2.remove_edge('A', 'D')
    G2.add_edge('A', 'Z')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    assert m == mapped
    G1.add_edges_from([(6, 6), (6, 6), (6, 6)])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert not m
    G2.add_edges_from([('E', 'E'), ('E', 'E'), ('E', 'E')])
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    assert m == mapped