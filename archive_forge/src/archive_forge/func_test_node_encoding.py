import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_node_encoding(self):
    """
        Test that tgrep search strings handles bytes and strs the same
        way.
        """
    tree = ParentedTree.fromstring('(S (NP (DT the) (JJ big) (NN dog)) (VP bit) (NP (DT a) (NN cat)))')
    self.assertEqual(list(tgrep.tgrep_positions(b'NN', [tree])), list(tgrep.tgrep_positions(b'NN', [tree])))
    self.assertEqual(list(tgrep.tgrep_nodes(b'NN', [tree])), list(tgrep.tgrep_nodes('NN', [tree])))
    self.assertEqual(list(tgrep.tgrep_positions(b'NN|JJ', [tree])), list(tgrep.tgrep_positions('NN|JJ', [tree])))