import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_node_noleaves(self):
    """
        Test node name matching with the search_leaves flag set to False.
        """
    tree = ParentedTree.fromstring('(S (A (T x)) (B (N x)))')
    self.assertEqual(list(tgrep.tgrep_positions('x', [tree])), [[(0, 0, 0), (1, 0, 0)]])
    self.assertEqual(list(tgrep.tgrep_positions('x', [tree], False)), [[]])