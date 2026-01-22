import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_multiple_conjs(self):
    """
        Test that multiple (3 or more) conjunctions of node relations are
        handled properly.
        """
    sent = ParentedTree.fromstring('((A (B b) (C c)) (A (B b) (C c) (D d)))')
    self.assertEqual(list(tgrep.tgrep_positions('(A < B < C < D)', [sent])), [[(1,)]])
    self.assertEqual(list(tgrep.tgrep_positions('(A < B < C)', [sent])), [[(0,), (1,)]])