import unittest
import numpy as np
from shapely.geometry import box, MultiPolygon, Point
def test_shapely_xy_attr_contains(self):
    g = Point(0, 0).buffer(10.0)
    self.assertContainsResults(self.construct_torus(), *g.exterior.xy)