import numpy as np
from holoviews.core import BoundingBox
from holoviews.element.comparison import ComparisonTestCase
def test_float_heterogeneous_unequal2(self):
    try:
        self.assertEqual(np.float64(3.54), 3.5)
    except AssertionError as e:
        if not str(e).startswith('Floats not almost equal to 6 decimals'):
            raise self.failureException('Numpy float mismatch error not raised.')