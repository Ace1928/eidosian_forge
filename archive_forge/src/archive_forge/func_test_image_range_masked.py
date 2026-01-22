import numpy as np
import holoviews as hv
from holoviews.element import Curve, Image
from ..utils import LoggingComparisonTestCase
def test_image_range_masked(self):
    arr = np.random.rand(10, 10) - 0.5
    arr = np.ma.masked_where(arr <= 0, arr)
    rrange = Image(arr).range(2)
    self.assertEqual(rrange, (np.min(arr), np.max(arr)))