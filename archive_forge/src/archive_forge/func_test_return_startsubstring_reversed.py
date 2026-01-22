import json
import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring
def test_return_startsubstring_reversed(self):
    assert substring(self.line1, -1, -500).wkt == LineString([(1, 0), (0, 0)]).wkt
    assert substring(self.line3, 3.5, 0).wkt == LineString([(0, 3.5), (0, 3), (0, 2), (0, 1), (0, 0)]).wkt
    assert substring(self.line3, -1.5, -500).wkt == LineString([(0, 2.5), (0, 2), (0, 1), (0, 0)]).wkt
    assert substring(self.line1, -0.5, -1.1, True).wkt == LineString([(1.0, 0), (0, 0)]).wkt
    assert substring(self.line3, 0.5, 0, True).wkt == LineString([(0, 2.0), (0, 1), (0, 0)]).wkt
    assert substring(self.line3, -0.5, -1.1, True).wkt == LineString([(0, 2.0), (0, 1), (0, 0)]).wkt