import json
import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring
def test_return_midsubstring_reversed(self):
    assert substring(self.line1, 0.6, 0.5).wkt == LineString([(0.6, 0), (0.5, 0)]).wkt
    assert substring(self.line1, -0.5, -0.6).wkt == LineString([(1.5, 0), (1.4, 0)]).wkt
    assert substring(self.line1, 0.6, 0.5, True).wkt == LineString([(1.2, 0), (1, 0)]).wkt
    assert substring(self.line1, -0.5, -0.6, True).wkt == LineString([(1, 0), (0.8, 0)]).wkt
    assert substring(self.line3, 3.5, 2.5).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
    assert substring(self.line3, -0.5, -1.5).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
    assert substring(self.line3, 3.5, -1.5).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
    assert substring(self.line3, -0.5, 2.5).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
    assert substring(self.line3, 0.875, 0.625, True).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
    assert substring(self.line3, -0.125, -0.375, True).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
    assert substring(self.line3, 0.875, -0.375, True).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
    assert substring(self.line3, -0.125, 0.625, True).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt