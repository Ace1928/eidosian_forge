import unittest
from collections import Counter
from timeit import timeit
from nltk.lm import Vocabulary
def test_vocab_iter_respects_cutoff(self):
    vocab_counts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'w', 'z']
    vocab_items = ['a', 'b', 'd', 'e', '<UNK>']
    self.assertCountEqual(vocab_counts, list(self.vocab.counts.keys()))
    self.assertCountEqual(vocab_items, list(self.vocab))