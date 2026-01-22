import numpy as np
from holoviews import Bars, Curve, Dimension, Histogram, Points, Scatter, VectorField
from holoviews.element.comparison import ComparisonTestCase
def test_vfield_unequal_1(self):
    try:
        self.assertEqual(self.vfield1, self.vfield2)
    except AssertionError as e:
        if not str(e).startswith('VectorField not almost equal to 6 decimals'):
            raise self.failureException('VectorField  data mismatch error not raised.')