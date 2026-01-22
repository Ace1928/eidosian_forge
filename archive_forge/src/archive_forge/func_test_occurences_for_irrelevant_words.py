import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence.text_analysis import (
from gensim.test.utils import common_texts
def test_occurences_for_irrelevant_words(self):
    accumulator = self.init_accumulator().accumulate(self.texts, 2)
    with self.assertRaises(KeyError):
        accumulator.get_occurrences('irrelevant')
    with self.assertRaises(KeyError):
        accumulator.get_co_occurrences('test', 'irrelevant')