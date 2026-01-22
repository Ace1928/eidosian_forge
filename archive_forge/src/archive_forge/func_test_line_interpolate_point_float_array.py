import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_line_interpolate_point_float_array():
    actual = shapely.line_interpolate_point(line_string, [0.2, 1.5, -0.2])
    assert_geometries_equal(actual[0], Point(0.2, 0))
    assert_geometries_equal(actual[1], Point(1, 0.5))
    assert_geometries_equal(actual[2], Point(1, 0.8))