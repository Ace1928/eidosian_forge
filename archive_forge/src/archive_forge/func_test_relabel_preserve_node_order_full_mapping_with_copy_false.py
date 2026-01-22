import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_preserve_node_order_full_mapping_with_copy_false(self):
    G = nx.path_graph(3)
    original_order = list(G)
    mapping = {2: 'a', 1: 'b', 0: 'c'}
    H = nx.relabel_nodes(G, mapping, copy=False)
    new_order = list(H)
    assert [mapping.get(i, i) for i in original_order] == new_order