import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
def test_points_unequal_data_values(self):
    try:
        self.assertEqual(self.points1, self.points3)
    except AssertionError as e:
        if not str(e).startswith('Points not almost equal to 6 decimals'):
            raise self.failureException('Points data mismatch error not raised.')