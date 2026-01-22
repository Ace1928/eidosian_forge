from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_lca_error_if_input_not_tree(self):
    G = nx.DiGraph([(1, 2), (2, 1)])
    pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))
    G = nx.DiGraph([(0, 2), (1, 2)])
    pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))