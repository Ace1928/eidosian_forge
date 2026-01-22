import unittest
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, Point
def test_alias_project(self):
    assert self.line1.line_locate_point(self.point) == 1.0
    assert self.line1.line_locate_point(self.point, normalized=True) == 0.5