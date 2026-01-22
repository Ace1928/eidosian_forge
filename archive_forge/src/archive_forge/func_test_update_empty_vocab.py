import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_update_empty_vocab(self):
    empty = Vocabulary(unk_cutoff=2)
    self.assertEqual(len(empty), 0)
    self.assertFalse(empty)
    self.assertIn(empty.unk_label, empty)
    empty.update(list('abcde'))
    self.assertIn(empty.unk_label, empty)