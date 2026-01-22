import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
def test_disconnected_graph_some_same_labels(self):
    G1 = nx.Graph()
    G1.add_nodes_from(list(range(10)))
    mapped = {0: 9, 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 0}
    G2 = nx.relabel_nodes(G1, mapped)
    colors = ['white', 'white', 'white', 'purple', 'purple', 'red', 'red', 'pink', 'pink', 'pink']
    nx.set_node_attributes(G1, dict(zip(G1, it.cycle(colors))), 'label')
    nx.set_node_attributes(G2, dict(zip([mapped[n] for n in G1], it.cycle(colors))), 'label')
    assert vf2pp_isomorphism(G1, G2, node_label='label')