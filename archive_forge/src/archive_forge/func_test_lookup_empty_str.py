import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_lookup_empty_str(self):
    self.assertEqual(self.vocab.lookup(''), '<UNK>')