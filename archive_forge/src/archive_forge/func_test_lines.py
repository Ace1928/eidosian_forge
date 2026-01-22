import unittest
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import triangulate
def test_lines(self):
    polys = triangulate(self.p, edges=True)
    assert len(polys) == 5
    for p in polys:
        assert isinstance(p, LineString)