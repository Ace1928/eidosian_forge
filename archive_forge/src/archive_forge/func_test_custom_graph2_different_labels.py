import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_graph2_different_labels(self):
    G1 = nx.Graph()
    mapped = {1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'G', 7: 'B', 6: 'F'}
    edges1 = [(1, 2), (1, 5), (5, 6), (2, 3), (2, 4), (3, 4), (4, 5), (2, 7)]
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
    nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
    G1.add_node(0)
    G2.add_node('Z')
    G1.nodes[0]['label'] = G1.nodes[1]['label']
    G2.nodes['Z']['label'] = G1.nodes[1]['label']
    mapped.update({0: 'Z'})
    assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped
    G2.nodes['Z']['label'] = G1.nodes[2]['label']
    assert vf2pp_isomorphism(G1, G2, node_label='label') is None
    G1.nodes[0]['label'] = 'blue'
    G2.nodes['Z']['label'] = 'blue'
    G1.add_edge(0, 1)
    assert vf2pp_isomorphism(G1, G2, node_label='label') is None
    G2.add_edge('Z', 'A')
    assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped