from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_lca_nonexisting_pairs_exception(self):
    lca = tree_all_pairs_lca(self.DG, 0, [(-1, -1)])
    pytest.raises(nx.NodeNotFound, list, lca)
    lca = tree_all_pairs_lca(self.DG, None, [(-1, -1)])
    pytest.raises(nx.NodeNotFound, list, lca)