import unittest
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import dump_coords
from shapely.ops import polygonize, polygonize_full
def test_polygonize(self):
    lines = [LineString([(0, 0), (1, 1)]), LineString([(0, 0), (0, 1)]), LineString([(0, 1), (1, 1)]), LineString([(1, 1), (1, 0)]), LineString([(1, 0), (0, 0)]), LineString([(5, 5), (6, 6)]), Point(0, 0)]
    result = list(polygonize(lines))
    assert all((isinstance(x, Polygon) for x in result))