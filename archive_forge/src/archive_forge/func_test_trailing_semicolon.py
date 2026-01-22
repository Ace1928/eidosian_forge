import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_trailing_semicolon(self):
    """
        Test that semicolons at the end of a tgrep2 search string won't
        cause a parse failure.
        """
    tree = ParentedTree.fromstring('(S (NP (DT the) (JJ big) (NN dog)) (VP bit) (NP (DT a) (NN cat)))')
    self.assertEqual(list(tgrep.tgrep_positions('NN', [tree])), [[(0, 2), (2, 1)]])
    self.assertEqual(list(tgrep.tgrep_positions('NN;', [tree])), [[(0, 2), (2, 1)]])
    self.assertEqual(list(tgrep.tgrep_positions('NN;;', [tree])), [[(0, 2), (2, 1)]])