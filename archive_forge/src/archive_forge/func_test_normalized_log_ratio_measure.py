import logging
import unittest
from collections import namedtuple
from gensim.topic_coherence import direct_confirmation_measure
from gensim.topic_coherence import text_analysis
def test_normalized_log_ratio_measure(self):
    """Test normalized_log_ratio_measure()"""
    obtained = direct_confirmation_measure.log_ratio_measure(self.segmentation, self.accumulator, normalize=True)[0]
    expected = -0.113282753
    self.assertAlmostEqual(expected, obtained)
    mean, std = direct_confirmation_measure.log_ratio_measure(self.segmentation, self.accumulator, normalize=True, with_std=True)[0]
    self.assertAlmostEqual(expected, mean)
    self.assertEqual(0.0, std)