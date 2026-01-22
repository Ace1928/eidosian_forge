import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_points_ellipsis_slice_x(self):
    sliced = hv.Points([(i, 2 * i) for i in range(10)])[2:7, ...]
    self.assertEqual(sliced.range('x'), (2, 6))