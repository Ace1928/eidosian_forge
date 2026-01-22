import json
import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring
def test_return_midpoint(self):
    assert substring(self.line1, 0.5, 0.5).equals(Point(0.5, 0))
    assert substring(self.line1, -0.5, -0.5).equals(Point(1.5, 0))
    assert substring(self.line1, 0.5, 0.5, True).equals(Point(1, 0))
    assert substring(self.line1, -0.5, -0.5, True).equals(Point(1, 0))
    assert substring(self.line1, 1.5, -0.5).equals(Point(1.5, 0))
    assert substring(self.line1, -0.5, 1.5).equals(Point(1.5, 0))
    assert substring(self.line1, -0.7, 0.3, True).equals(Point(0.6, 0))
    assert substring(self.line1, 0.3, -0.7, True).equals(Point(0.6, 0))