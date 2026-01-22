import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
@staticmethod
def create_fake_language_model():
    language_prob = defaultdict(lambda: -999.0)
    language_prob['my',] = log(0.1)
    language_prob['hovercraft',] = log(0.1)
    language_prob['is',] = log(0.1)
    language_prob['full',] = log(0.1)
    language_prob['of',] = log(0.1)
    language_prob['eels',] = log(0.1)
    language_prob['my', 'hovercraft'] = log(0.3)
    language_model = type('', (object,), {'probability': lambda _, phrase: language_prob[phrase]})()
    return language_model