import logging
import unittest
import numpy as np
from gensim.test.utils import datapath
from gensim.models.keyedvectors import KeyedVectors
def test_high_precision(self):
    kv = self.load_model(np.float64)
    self.assertAlmostEqual(kv['horse.n.01'][0], -0.0008546282343595379)
    self.assertEqual(kv['horse.n.01'][0].dtype, np.float64)