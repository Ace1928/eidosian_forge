import logging
import unittest
from collections import namedtuple
from gensim.topic_coherence import direct_confirmation_measure
from gensim.topic_coherence import text_analysis
def test_log_conditional_probability(self):
    """Test log_conditional_probability()"""
    obtained = direct_confirmation_measure.log_conditional_probability(self.segmentation, self.accumulator)[0]
    expected = -0.693147181
    self.assertAlmostEqual(expected, obtained)
    mean, std = direct_confirmation_measure.log_conditional_probability(self.segmentation, self.accumulator, with_std=True)[0]
    self.assertAlmostEqual(expected, mean)
    self.assertEqual(0.0, std)