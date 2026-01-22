import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_similar_by_word(self):
    """Test similar_by_word returns expected results."""
    expected = ['conflict', 'administration', 'terrorism', 'call', 'israel']
    predicted = [result[0] for result in self.vectors.similar_by_word('war', topn=5)]
    self.assertEqual(expected, predicted)