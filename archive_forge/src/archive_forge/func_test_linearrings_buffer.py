import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_linearrings_buffer(dim, order):
    coords = np.random.randn(10, 4, dim)
    coords1 = np.asarray(coords.reshape(10 * 4, dim), order=order)
    indices1 = np.repeat(range(10), 4)
    result1 = shapely.linearrings(coords1, indices=indices1)
    coords2 = np.hstack((coords, coords[:, [0], :]))
    coords2 = np.asarray(coords2.reshape(10 * 5, dim), order=order)
    indices2 = np.repeat(range(10), 5)
    result2 = shapely.linearrings(coords2, indices=indices2)
    assert_geometries_equal(result1, result2)