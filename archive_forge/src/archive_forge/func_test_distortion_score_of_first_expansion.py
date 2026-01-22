import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_distortion_score_of_first_expansion(self):
    stack_decoder = StackDecoder(None, None)
    stack_decoder.distortion_factor = 0.5
    hypothesis = _Hypothesis()
    score = stack_decoder.distortion_score(hypothesis, (8, 10))
    self.assertEqual(score, 0.0)