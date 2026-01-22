import unittest
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import triangulate
def test_polys(self):
    polys = triangulate(self.p)
    assert len(polys) == 2
    for p in polys:
        assert isinstance(p, Polygon)