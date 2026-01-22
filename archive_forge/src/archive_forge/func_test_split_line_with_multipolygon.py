import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
def test_split_line_with_multipolygon(self):
    poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    poly2 = Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5), (0.5, 0.5)])
    poly3 = Polygon([(0, 0), (0, -2), (-2, -2), (-2, 0), (0, 0)])
    splitter = MultiPolygon([poly1, poly2, poly3])
    self.helper(self.ls, splitter, 4)