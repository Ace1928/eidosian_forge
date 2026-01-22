import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence.text_analysis import (
from gensim.test.utils import common_texts
def test_occurrence_counting(self):
    accumulator = self.init_accumulator().accumulate(self.texts, 3)
    self.assertEqual(3, accumulator.get_occurrences('this'))
    self.assertEqual(1, accumulator.get_occurrences('is'))
    self.assertEqual(1, accumulator.get_occurrences('a'))
    self.assertEqual(2, accumulator.get_co_occurrences('test', 'document'))
    self.assertEqual(2, accumulator.get_co_occurrences('test', 'this'))
    self.assertEqual(1, accumulator.get_co_occurrences('is', 'a'))