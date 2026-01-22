import unittest
import numpy as np
from shapely.geometry import box, MultiPolygon, Point
def test_xy_array_order(self):
    y, x = np.mgrid[-10:10:5j, -5:15:5j]
    x = x.copy('f')
    y = y.copy('f')
    result = self.assertContainsResults(self.construct_torus(), x, y)
    assert result.flags['F_CONTIGUOUS']