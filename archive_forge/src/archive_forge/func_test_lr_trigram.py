import unittest
from nltk.metrics import (
def test_lr_trigram(self):
    self.assertAlmostEqual(TrigramAssocMeasures.likelihood_ratio(1, (1, 1, 1), (1, 1, 1), 2), 5.545177444479562, delta=_DELTA)
    self.assertAlmostEqual(TrigramAssocMeasures.likelihood_ratio(1, (1, 1, 1), (1, 1, 1), 1), 0.0, delta=_DELTA)
    self.assertRaises(ValueError, TrigramAssocMeasures.likelihood_ratio, *(1, (1, 1, 2), (1, 1, 2), 2))