import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('geom', [line_string, linear_ring])
def test_get_point(geom):
    n = shapely.get_num_points(geom)
    actual = shapely.get_point(geom, [0, -n, n, -(n + 1)])
    assert_geometries_equal(actual[0], actual[1])
    assert shapely.is_missing(actual[2:4]).all()