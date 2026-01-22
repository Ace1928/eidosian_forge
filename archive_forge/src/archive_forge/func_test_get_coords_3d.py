import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
@pytest.mark.parametrize('include_z', [True, False])
@pytest.mark.parametrize('geoms,x,y,z', [([point, point_z], [2, 2], [3, 3], [np.nan, 4]), ([line_string_z], [0, 1, 1], [0, 0, 1], [4, 4, 4]), ([polygon_z], [0, 2, 2, 0, 0], [0, 0, 2, 2, 0], [4, 4, 4, 4, 4]), ([geometry_collection_z], [2, 0, 1, 1], [3, 0, 0, 1], [4, 4, 4, 4]), ([point, empty_point], [2], [3], [np.nan])])
def test_get_coords_3d(geoms, x, y, z, include_z):
    actual = get_coordinates(np.array(geoms, np.object_), include_z=include_z)
    if include_z:
        expected = np.array([x, y, z], np.float64).T
    else:
        expected = np.array([x, y], np.float64).T
    assert_equal(actual, expected)