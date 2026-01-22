import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_future_score(self):
    hypothesis = _Hypothesis()
    hypothesis.untranslated_spans = lambda _: [(0, 2), (5, 8)]
    future_score_table = defaultdict(lambda: defaultdict(float))
    future_score_table[0][2] = 0.4
    future_score_table[5][8] = 0.5
    stack_decoder = StackDecoder(None, None)
    future_score = stack_decoder.future_score(hypothesis, future_score_table, 8)
    self.assertEqual(future_score, 0.4 + 0.5)