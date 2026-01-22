import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('coordinates', [[[1, 1], [2, 1], [2, 2], [1, 1]], [[1, 1], [2, 1], [2, 2]]])
def test_linearrings(coordinates):
    actual = shapely.linearrings(np.array(coordinates, dtype=np.float64), indices=np.zeros(len(coordinates), dtype=np.intp))
    assert_geometries_equal(actual, shapely.linearrings(coordinates))