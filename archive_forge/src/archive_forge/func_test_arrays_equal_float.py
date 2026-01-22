import numpy as np
from holoviews.core import BoundingBox
from holoviews.element.comparison import ComparisonTestCase
def test_arrays_equal_float(self):
    self.assertEqual(np.array([[1.0, 2.5], [3, 4]], dtype=np.float32), np.array([[1.0, 2.5], [3, 4]], dtype=np.float32))