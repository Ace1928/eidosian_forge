import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_multigraph2_same_labels(self):
    G1 = nx.MultiGraph()
    mapped = {1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'G', 7: 'B', 6: 'F'}
    edges1 = [(1, 2), (1, 2), (1, 5), (1, 5), (1, 5), (5, 6), (2, 3), (2, 3), (2, 4), (3, 4), (3, 4), (4, 5), (4, 5), (4, 5), (2, 7), (2, 7), (2, 7)]
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_same))), 'label')
    nx.set_node_attributes(G2, dict(zip(G2, it.cycle(labels_same))), 'label')
    m = vf2pp_isomorphism(G1, G2, node_label='label')
    assert m
    G2.remove_edges_from([(mapped[1], mapped[2]), (mapped[1], mapped[2])])
    G2.add_edge(mapped[1], mapped[4])
    H1 = nx.MultiGraph(G1.subgraph([2, 3, 4, 7]))
    H2 = nx.MultiGraph(G2.subgraph([mapped[1], mapped[4], mapped[5], mapped[6]]))
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert not m
    H1.remove_edge(3, 4)
    H1.add_edges_from([(2, 3), (2, 4), (2, 4)])
    H2.add_edges_from([(mapped[5], mapped[6]), (mapped[5], mapped[6])])
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m
    H1.remove_edges_from([(2, 3), (2, 3), (2, 3)])
    H2.remove_edges_from([(mapped[5], mapped[4])] * 3)
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m
    H1.remove_edges_from([(2, 7), (2, 7)])
    H1.add_edges_from([(3, 4), (3, 4)])
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert m
    H2.add_edge(mapped[5], mapped[1])
    H1.add_edge(3, 4)
    m = vf2pp_isomorphism(H1, H2, node_label='label')
    assert not m