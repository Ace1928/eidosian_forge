import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_distortion_score(self):
    stack_decoder = StackDecoder(None, None)
    stack_decoder.distortion_factor = 0.5
    hypothesis = _Hypothesis()
    hypothesis.src_phrase_span = (3, 5)
    score = stack_decoder.distortion_score(hypothesis, (8, 10))
    expected_score = log(stack_decoder.distortion_factor) * (8 - 5)
    self.assertEqual(score, expected_score)