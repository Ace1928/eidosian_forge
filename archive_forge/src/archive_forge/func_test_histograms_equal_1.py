import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
def test_histograms_equal_1(self):
    self.assertEqual(self.hist1, self.hist1)