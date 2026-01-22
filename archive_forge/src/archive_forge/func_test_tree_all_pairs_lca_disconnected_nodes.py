from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_lca_disconnected_nodes(self):
    G = nx.DiGraph()
    G.add_node(1)
    assert {(1, 1): 1} == dict(tree_all_pairs_lca(G))
    G.add_node(0)
    assert {(1, 1): 1} == dict(tree_all_pairs_lca(G, 1))
    assert {(0, 0): 0} == dict(tree_all_pairs_lca(G, 0))
    pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))