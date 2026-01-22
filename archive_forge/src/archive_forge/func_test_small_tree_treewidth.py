import itertools
import networkx as nx
from networkx.algorithms.approximation import (
from networkx.algorithms.approximation.treewidth import (
def test_small_tree_treewidth(self):
    """Test if the computed treewidth of the known self.small_tree is 2"""
    G = self.small_tree
    treewidth, _ = treewidth_min_fill_in(G)
    assert treewidth == 2