import functools
import logging
import unittest
import numpy as np
from gensim.models.keyedvectors import KeyedVectors, REAL, pseudorandom_weak_vector
from gensim.test.utils import datapath
import gensim.models.keyedvectors
def test_most_similar_with_vector_input(self):
    """Test most_similar returns expected results with an input vector instead of an input word."""
    expected = ['war', 'conflict', 'administration', 'terrorism', 'call']
    input_vector = self.vectors['war']
    predicted = [result[0] for result in self.vectors.most_similar([input_vector], topn=5)]
    self.assertEqual(expected, predicted)