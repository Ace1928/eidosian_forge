import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
@pytest.mark.parametrize('geom,expected', [(point, [2, 3, 2, 3]), (shapely.linestrings([[0, 0], [0, 1]]), [0, 0, 0, 1]), (shapely.linestrings([[0, 0], [1, 0]]), [0, 0, 1, 0]), (multi_point, [0, 0, 1, 2]), (multi_polygon, [0, 0, 2.2, 2.2]), (geometry_collection, [49, -1, 52, 2]), (empty, [np.nan, np.nan, np.nan, np.nan]), (None, [np.nan, np.nan, np.nan, np.nan]), ([empty, empty, None], [np.nan, np.nan, np.nan, np.nan]), ([point, None], [2, 3, 2, 3]), ([point, empty], [2, 3, 2, 3]), ([point, empty, None], [2, 3, 2, 3]), ([point, empty, None, multi_point], [0, 0, 2, 3])])
def test_total_bounds(geom, expected):
    assert_array_equal(shapely.total_bounds(geom), expected)