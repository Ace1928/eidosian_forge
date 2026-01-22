import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('geometries,indices,expected', [([point, line_string], [0, 0], [geom_coll([point, line_string])]), ([point, line_string], [0, 1], [geom_coll([point]), geom_coll([line_string])]), ([point, None], [0, 0], [geom_coll([point])]), ([point, None], [0, 1], [geom_coll([point]), geom_coll([])]), ([None, point, None, None], [0, 0, 1, 1], [geom_coll([point]), geom_coll([])]), ([point, None, line_string], [0, 0, 0], [geom_coll([point, line_string])])])
def test_geometrycollections(geometries, indices, expected):
    actual = shapely.geometrycollections(np.array(geometries, dtype=object), indices=indices)
    assert_geometries_equal(actual, expected)