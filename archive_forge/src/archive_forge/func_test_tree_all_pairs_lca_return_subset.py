from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_lca_return_subset(self):
    test_pairs = [(0, 1), (0, 1), (1, 0)]
    ans = dict(tree_all_pairs_lca(self.DG, 0, test_pairs))
    assert (0, 1) in ans and (1, 0) in ans
    assert len(ans) == 2