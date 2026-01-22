import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
def test_empty_references(self):
    references = [[]]
    hypothesis = 'John loves Mary'.split()
    assert sentence_bleu(references, hypothesis) == 0