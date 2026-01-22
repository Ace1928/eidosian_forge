import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_preserve_node_order_partial_mapping_with_copy_true(self):
    G = nx.path_graph(3)
    original_order = list(G)
    mapping = {1: 'a', 0: 'b'}
    H = nx.relabel_nodes(G, mapping, copy=True)
    new_order = list(H)
    assert [mapping.get(i, i) for i in original_order] == new_order