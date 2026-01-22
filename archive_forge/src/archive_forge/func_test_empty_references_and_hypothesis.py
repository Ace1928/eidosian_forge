import io
import unittest
from nltk.data import find
from nltk.translate.bleu_score import (
def test_empty_references_and_hypothesis(self):
    references = [[]]
    hypothesis = []
    assert sentence_bleu(references, hypothesis) == 0