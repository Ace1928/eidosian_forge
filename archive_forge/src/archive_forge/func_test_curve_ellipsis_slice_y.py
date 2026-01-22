import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_curve_ellipsis_slice_y(self):
    sliced = hv.Curve([(i, 2 * i) for i in range(10)])[..., 3:9]
    self.assertEqual(sliced.range('y'), (4, 8))