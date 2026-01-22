import unittest
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point
def test_line_ends_interpolate(self):
    assert self.line1.interpolate(-1000).equals(Point(0.0, 0.0))
    assert self.line1.interpolate(1000).equals(Point(2.0, 0.0))