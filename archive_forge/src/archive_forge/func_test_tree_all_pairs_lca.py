from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_lca(self):
    all_pairs = chain(combinations(self.DG, 2), ((node, node) for node in self.DG))
    ans = dict(tree_all_pairs_lca(self.DG, 0, all_pairs))
    self.assert_has_same_pairs(ans, self.ans)