import unittest
from collections import defaultdict
from math import log
from nltk.translate import PhraseTable, StackDecoder
from nltk.translate.stack_decoder import _Hypothesis, _Stack
def test_compute_future_costs(self):
    phrase_table = TestStackDecoder.create_fake_phrase_table()
    language_model = TestStackDecoder.create_fake_language_model()
    stack_decoder = StackDecoder(phrase_table, language_model)
    sentence = ('my', 'hovercraft', 'is', 'full', 'of', 'eels')
    future_scores = stack_decoder.compute_future_scores(sentence)
    self.assertEqual(future_scores[1][2], phrase_table.translations_for(('hovercraft',))[0].log_prob + language_model.probability(('hovercraft',)))
    self.assertEqual(future_scores[0][2], phrase_table.translations_for(('my', 'hovercraft'))[0].log_prob + language_model.probability(('my', 'hovercraft')))