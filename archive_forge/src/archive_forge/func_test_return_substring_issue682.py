import json
import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring
def test_return_substring_issue682(self):
    assert list(substring(self.line2, 0.1, 0).coords) == [(3.0, 0.1), (3.0, 0.0)]