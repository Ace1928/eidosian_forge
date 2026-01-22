from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_lca_default_root(self):
    assert dict(tree_all_pairs_lca(self.DG)) == self.ans