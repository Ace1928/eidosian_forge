from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_all_pairs_lca_gold_example(self):
    self.assert_lca_dicts_same(dict(all_pairs_lca(self.DG)), self.gold)