import unittest
import numpy as np
from shapely.geometry import box, MultiPolygon, Point
def test_array_2d(self):
    y, x = np.mgrid[-10:10:15j, -5:15:16j]
    result = self.assertContainsResults(self.construct_torus(), x, y)
    assert result.shape == x.shape