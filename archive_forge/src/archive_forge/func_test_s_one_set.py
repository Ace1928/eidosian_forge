import logging
import unittest
import numpy as np
from gensim.topic_coherence import segmentation
from numpy import array
def test_s_one_set(self):
    """Test s_one_set segmentation."""
    actual = segmentation.s_one_set(self.topics)
    expected = [[(9, array([9, 4, 6])), (4, array([9, 4, 6])), (6, array([9, 4, 6]))], [(9, array([9, 10, 7])), (10, array([9, 10, 7])), (7, array([9, 10, 7]))], [(5, array([5, 2, 7])), (2, array([5, 2, 7])), (7, array([5, 2, 7]))]]
    for s_i in range(len(actual)):
        for j in range(len(actual[s_i])):
            self.assertEqual(actual[s_i][j][0], expected[s_i][j][0])
            self.assertTrue(np.allclose(actual[s_i][j][1], expected[s_i][j][1]))