import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
def test_histograms_unequal_2(self):
    with self.assertRaises(AssertionError):
        self.assertEqual(self.hist1, self.hist3)