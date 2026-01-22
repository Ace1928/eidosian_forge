import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence.text_analysis import (
from gensim.test.utils import common_texts
def test_accumulate2(self):
    accumulator = InvertedIndexAccumulator(self.top_ids, self.dictionary).accumulate(self.texts, 3)
    inverted_index = accumulator.index_to_dict()
    expected = {10: {0, 2, 3}, 15: {0}, 20: {0}, 21: {1, 2, 3}, 17: {1, 2}}
    self.assertDictEqual(expected, inverted_index)