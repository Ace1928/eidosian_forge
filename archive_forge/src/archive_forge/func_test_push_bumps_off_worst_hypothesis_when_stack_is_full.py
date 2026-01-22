import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_push_bumps_off_worst_hypothesis_when_stack_is_full(self):
    stack = _Stack(3)
    poor_hypothesis = _Hypothesis(0.01)
    stack.push(_Hypothesis(0.2))
    stack.push(poor_hypothesis)
    stack.push(_Hypothesis(0.1))
    stack.push(_Hypothesis(0.3))
    self.assertFalse(poor_hypothesis in stack)