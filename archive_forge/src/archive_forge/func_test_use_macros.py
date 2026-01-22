import unittest
from nltk import tgrep
from nltk.tree import ParentedTree
def test_use_macros(self):
    """
        Test defining and using tgrep2 macros.
        """
    tree = ParentedTree.fromstring('(VP (VB sold) (NP (DET the) (NN heiress)) (NP (NN deed) (PREP to) (NP (DET the) (NN school) (NN house))))')
    self.assertEqual(list(tgrep.tgrep_positions('@ NP /^NP/;\n@ NN /^NN/;\n@NP !< @NP !$.. @NN', [tree])), [[(1,), (2, 2)]])
    self.assertRaises(tgrep.TgrepException, list, tgrep.tgrep_positions('@ NP /^NP/;\n@ NN /^NN/;\n@CNP !< @NP !$.. @NN', [tree]))