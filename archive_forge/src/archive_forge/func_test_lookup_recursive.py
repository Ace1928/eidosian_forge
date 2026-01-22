import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_lookup_recursive(self):
    self.assertEqual(self.vocab.lookup([['a', 'b'], ['a', 'c']]), (('a', 'b'), ('a', '<UNK>')))
    self.assertEqual(self.vocab.lookup([['a', 'b'], 'c']), (('a', 'b'), '<UNK>'))
    self.assertEqual(self.vocab.lookup([[[[['a', 'b']]]]]), ((((('a', 'b'),),),),))