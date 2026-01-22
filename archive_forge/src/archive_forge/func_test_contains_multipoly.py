import unittest
import numpy as np
from shapely.geometry import box, MultiPolygon, Point
def test_contains_multipoly(self):
    y, x = (np.mgrid[-10:10:5j], np.mgrid[-5:15:5j])
    cut_poly = box(-1, -10, -2.5, 10)
    geom = self.construct_torus().difference(cut_poly)
    assert isinstance(geom, MultiPolygon)
    self.assertContainsResults(geom, x, y)