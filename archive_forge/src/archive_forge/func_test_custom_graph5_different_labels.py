import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_custom_graph5_different_labels(self):
    G1 = nx.Graph()
    edges1 = [(1, 5), (1, 2), (1, 4), (2, 3), (2, 6), (3, 4), (3, 7), (4, 8), (5, 8), (5, 6), (6, 7), (7, 8)]
    mapped = {1: 'a', 2: 'h', 3: 'd', 4: 'i', 5: 'g', 6: 'b', 7: 'j', 8: 'c'}
    G1.add_edges_from(edges1)
    G2 = nx.relabel_nodes(G1, mapped)
    colors = ['red', 'blue', 'grey', 'none', 'brown', 'solarized', 'yellow', 'pink']
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(labels_many))), 'label')
    nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(labels_many))), 'label')
    assert vf2pp_isomorphism(G1, G2, node_label='label') == mapped
    c = 0
    for node in G1.nodes():
        color1 = colors[c]
        color2 = colors[(c + 3) % len(colors)]
        G1.nodes[node]['label'] = color1
        G2.nodes[mapped[node]]['label'] = color2
        c += 1
    assert vf2pp_isomorphism(G1, G2, node_label='label') is None
    H1 = G1.subgraph([1, 5])
    H2 = G2.subgraph(['i', 'c'])
    c = 0
    for node1, node2 in zip(H1.nodes(), H2.nodes()):
        H1.nodes[node1]['label'] = 'red'
        H2.nodes[node2]['label'] = 'red'
        c += 1
    assert vf2pp_isomorphism(H1, H2, node_label='label')