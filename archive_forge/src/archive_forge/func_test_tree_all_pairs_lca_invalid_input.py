from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_lca_invalid_input(self):
    empty_digraph = tree_all_pairs_lca(nx.DiGraph())
    pytest.raises(nx.NetworkXPointlessConcept, list, empty_digraph)
    bad_pairs_digraph = tree_all_pairs_lca(self.DG, pairs=[(-1, -2)])
    pytest.raises(nx.NodeNotFound, list, bad_pairs_digraph)