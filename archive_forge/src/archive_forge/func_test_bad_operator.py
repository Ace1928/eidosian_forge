import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_bad_operator(self):
    """
        Test error handling of undefined tgrep operators.
        """
    tree = ParentedTree.fromstring('(S (A (T x)) (B (N x)))')
    self.assertRaises(tgrep.TgrepException, list, tgrep.tgrep_positions('* >>> S', [tree]))