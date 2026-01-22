import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_graph1_same_labels(self):
    G1 = nx.Graph()
    mapped = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'Z', 6: 'E'}
    edges1 = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 6), (3, 4), (5, 1), (5, 2)]
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
    assert vf2pp_isomorphism(G1, G2, node_label='label')
    G1.add_edge(3, 7)
    G1.nodes[7]['label'] = 'blue'
    assert vf2pp_isomorphism(G1, G2, node_label='label') is None
    G2.add_edges_from([(mapped[3], 'X'), (mapped[6], mapped[5])])
    G1.add_edge(4, 7)
    G2.nodes['X']['label'] = 'blue'
    assert vf2pp_isomorphism(G1, G2, node_label='label')
    G1.remove_edges_from([(1, 4), (1, 3)])
    G2.remove_edges_from([(mapped[1], mapped[5]), (mapped[1], mapped[2])])
    assert vf2pp_isomorphism(G1, G2, node_label='label')