import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
def test_get_parts_array():
    geom = np.array([None, empty_line_string, multi_point, point, multi_polygon])
    expected_parts = []
    for g in geom:
        for i in range(0, shapely.get_num_geometries(g)):
            expected_parts.append(shapely.get_geometry(g, i))
    parts = shapely.get_parts(geom)
    assert len(parts) == len(expected_parts)
    assert_geometries_equal(parts, expected_parts)