import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_vocab_len_respects_cutoff(self):
    self.assertEqual(5, len(self.vocab))