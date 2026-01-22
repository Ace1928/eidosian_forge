import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_line_locate_point_geom_array2():
    points = shapely.points([[0, 0], [1, 0]])
    actual = shapely.line_locate_point(line_string, points)
    np.testing.assert_allclose(actual, [0.0, 1.0])