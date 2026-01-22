import json
import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring
def test_return_endsubstring(self):
    assert substring(self.line1, 0.6, 500).wkt == LineString([(0.6, 0), (2, 0)]).wkt
    assert substring(self.line1, 0.6, 1.1, True).wkt == LineString([(1.2, 0), (2, 0)]).wkt