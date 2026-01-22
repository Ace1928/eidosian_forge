import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('coordinates,indices,expected', [([[1, 1], [2, 2]], [0, 0], [lstrs([[1, 1], [2, 2]])]), ([[1, 1, 1], [2, 2, 2]], [0, 0], [lstrs([[1, 1, 1], [2, 2, 2]])]), ([[1, 1], [2, 2], [2, 2], [3, 3]], [0, 0, 1, 1], [lstrs([[1, 1], [2, 2]]), lstrs([[2, 2], [3, 3]])])])
def test_linestrings(coordinates, indices, expected):
    actual = shapely.linestrings(np.array(coordinates, dtype=float), indices=np.array(indices, dtype=np.intp))
    assert_geometries_equal(actual, expected)