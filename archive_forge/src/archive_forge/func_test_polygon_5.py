import unittest
from shapely.geometry import Polygon
def test_polygon_5(self):
    p = (1.0, 1.0)
    poly = Polygon([p, p, p, p, p])
    assert poly.bounds == (1.0, 1.0, 1.0, 1.0)