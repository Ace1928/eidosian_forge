import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
def test_scatter_unequal_data_shape(self):
    try:
        self.assertEqual(self.scatter1, self.scatter2)
    except AssertionError as e:
        if not str(e).startswith('Scatter not of matching length, 20 vs. 21.'):
            raise self.failureException('Scatter data mismatch error not raised.')