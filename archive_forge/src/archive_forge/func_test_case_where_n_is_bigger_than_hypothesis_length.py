import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
def test_case_where_n_is_bigger_than_hypothesis_length(self):
    references = ['John loves Mary ?'.split()]
    hypothesis = 'John loves Mary'.split()
    n = len(hypothesis) + 1
    weights = (1.0 / n,) * n
    self.assertAlmostEqual(sentence_bleu(references, hypothesis, weights), 0.0, places=4)
    try:
        self.assertWarns(UserWarning, sentence_bleu, references, hypothesis)
    except AttributeError:
        pass
    references = ['John loves Mary'.split()]
    hypothesis = 'John loves Mary'.split()
    self.assertAlmostEqual(sentence_bleu(references, hypothesis, weights), 0.0, places=4)