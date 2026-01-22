import unittest
from nltk.metrics import (
def test_lr_quadgram(self):
    self.assertAlmostEqual(QuadgramAssocMeasures.likelihood_ratio(1, (1, 1, 1, 1), (1, 1, 1, 1, 1, 1), (1, 1, 1, 1), 2), 8.317766166719343, delta=_DELTA)
    self.assertAlmostEqual(QuadgramAssocMeasures.likelihood_ratio(1, (1, 1, 1, 1), (1, 1, 1, 1, 1, 1), (1, 1, 1, 1), 1), 0.0, delta=_DELTA)
    self.assertRaises(ValueError, QuadgramAssocMeasures.likelihood_ratio, *(1, (1, 1, 1, 1), (1, 1, 1, 1, 1, 2), (1, 1, 1, 1), 1))