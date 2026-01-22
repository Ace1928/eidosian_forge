import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_node_regex_2(self):
    """
        Test regex matching on nodes.
        """
    tree = ParentedTree.fromstring('(S (SBJ x) (SBJ1 x) (NP-SBJ x))')
    self.assertEqual(list(tgrep.tgrep_positions('/^SBJ/', [tree])), [[(0,), (1,)]])
    self.assertEqual(list(tgrep.tgrep_positions('/SBJ/', [tree])), [[(0,), (1,), (2,)]])