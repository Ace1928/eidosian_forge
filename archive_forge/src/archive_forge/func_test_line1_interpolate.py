import unittest
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point
def test_line1_interpolate(self):
    assert self.line1.interpolate(0.5).equals(Point(0.5, 0.0))
    assert self.line1.interpolate(-0.5).equals(Point(1.5, 0.0))
    assert self.line1.interpolate(0.5, normalized=True).equals(Point(1, 0))
    assert self.line1.interpolate(-0.5, normalized=True).equals(Point(1, 0))