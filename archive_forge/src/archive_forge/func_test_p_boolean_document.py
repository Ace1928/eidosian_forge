import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.hashdictionary import HashDictionary
from gensim.topic_coherence import probability_estimation
def test_p_boolean_document(self):
    """Test p_boolean_document()"""
    accumulator = probability_estimation.p_boolean_document(self.corpus, self.segmented_topics)
    obtained = accumulator.index_to_dict()
    expected = {self.graph_id: {5}, self.user_id: {1, 3}, self.system_id: {1, 2}, self.computer_id: {0}}
    self.assertEqual(expected, obtained)