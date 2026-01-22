import logging
import unittest
from gensim.topic_coherence import aggregation
def test_arithmetic_mean(self):
    """Test arithmetic_mean()"""
    obtained = aggregation.arithmetic_mean(self.confirmed_measures)
    expected = 2.75
    self.assertEqual(obtained, expected)