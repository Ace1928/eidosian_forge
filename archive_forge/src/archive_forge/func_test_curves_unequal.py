import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
def test_curves_unequal(self):
    try:
        self.assertEqual(self.curve1, self.curve2)
    except AssertionError as e:
        if not str(e).startswith('Curve not of matching length, 100 vs. 101'):
            raise self.failureException('Curve mismatch error not raised.')