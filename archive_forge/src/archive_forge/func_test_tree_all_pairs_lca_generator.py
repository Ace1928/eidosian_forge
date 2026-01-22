from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_lca_generator(self):
    pairs = iter([(0, 1), (0, 1), (1, 0)])
    some_pairs = dict(tree_all_pairs_lca(self.DG, 0, pairs))
    assert (0, 1) in some_pairs and (1, 0) in some_pairs
    assert len(some_pairs) == 2