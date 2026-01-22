import json
import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring
def test_return_z_coord_issue1699(self):
    line_z = LineString([(0, 0, 0), (2, 0, 0)])
    assert substring(line_z, 0, 0.5, True).wkt == LineString([(0, 0, 0), (1, 0, 0)]).wkt
    assert substring(line_z, 0.5, 0, True).wkt == LineString([(1, 0, 0), (0, 0, 0)]).wkt