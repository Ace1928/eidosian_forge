import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_similar_by_vector(self):
    """Test similar_by_word returns expected results."""
    expected = ['war', 'conflict', 'administration', 'terrorism', 'call']
    input_vector = self.vectors['war']
    predicted = [result[0] for result in self.vectors.similar_by_vector(input_vector, topn=5)]
    self.assertEqual(expected, predicted)