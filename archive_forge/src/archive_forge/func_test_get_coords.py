import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
@pytest.mark.parametrize('include_z', [True, False])
@pytest.mark.parametrize('geoms,x,y', [([], [], []), ([empty], [], []), ([point, empty], [2], [3]), ([empty, point, empty], [2], [3]), ([point, None], [2], [3]), ([None, point, None], [2], [3]), ([point, point], [2, 2], [3, 3]), ([line_string, linear_ring], [0, 1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 0]), ([polygon], [0, 2, 2, 0, 0], [0, 0, 2, 2, 0]), ([polygon_with_hole], [0, 0, 10, 10, 0, 2, 2, 4, 4, 2], [0, 10, 10, 0, 0, 2, 4, 4, 2, 2]), ([multi_point, multi_line_string], [0, 1, 0, 1], [0, 2, 0, 2]), ([multi_polygon], [0, 1, 1, 0, 0, 2.1, 2.2, 2.2, 2.1, 2.1], [0, 0, 1, 1, 0, 2.1, 2.1, 2.2, 2.2, 2.1]), ([geometry_collection], [51, 52, 49], [-1, -1, 2]), ([nested_2], [51, 52, 49, 2], [-1, -1, 2, 3]), ([nested_3], [51, 52, 49, 2, 2], [-1, -1, 2, 3, 3])])
def test_get_coords(geoms, x, y, include_z):
    actual = get_coordinates(np.array(geoms, np.object_), include_z=include_z)
    if not include_z:
        expected = np.array([x, y], np.float64).T
    else:
        expected = np.array([x, y, [np.nan] * len(x)], np.float64).T
    assert_equal(actual, expected)