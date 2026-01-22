import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence.text_analysis import (
from gensim.test.utils import common_texts
def test_occurrence_counting2(self):
    accumulator = self.init_accumulator2().accumulate(self.texts2, 110)
    self.assertEqual(2, accumulator.get_occurrences('human'))
    self.assertEqual(4, accumulator.get_occurrences('user'))
    self.assertEqual(3, accumulator.get_occurrences('graph'))
    self.assertEqual(3, accumulator.get_occurrences('trees'))
    cases = [(1, ('human', 'interface')), (2, ('system', 'user')), (2, ('graph', 'minors')), (2, ('graph', 'trees')), (4, ('user', 'user')), (3, ('graph', 'graph')), (0, ('time', 'eps'))]
    for expected_count, (word1, word2) in cases:
        self.assertEqual(expected_count, accumulator.get_co_occurrences(word1, word2))
        self.assertEqual(expected_count, accumulator.get_co_occurrences(word2, word1))
        word_id1 = self.dictionary2.token2id[word1]
        word_id2 = self.dictionary2.token2id[word2]
        self.assertEqual(expected_count, accumulator.get_co_occurrences(word_id1, word_id2))
        self.assertEqual(expected_count, accumulator.get_co_occurrences(word_id2, word_id1))