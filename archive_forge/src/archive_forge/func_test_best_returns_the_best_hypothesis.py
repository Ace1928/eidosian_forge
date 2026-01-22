import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_best_returns_the_best_hypothesis(self):
    stack = _Stack(3)
    best_hypothesis = _Hypothesis(0.99)
    stack.push(_Hypothesis(0.0))
    stack.push(best_hypothesis)
    stack.push(_Hypothesis(0.5))
    self.assertEqual(stack.best(), best_hypothesis)