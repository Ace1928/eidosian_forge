import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_line_merge_geom_array():
    actual = shapely.line_merge([line_string, multi_line_string])
    assert_geometries_equal(actual[0], line_string)
    assert_geometries_equal(actual[1], LineString([(0, 0), (1, 2)]))