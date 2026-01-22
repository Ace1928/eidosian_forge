import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
@pytest.mark.parametrize('geoms,index', [([], []), ([empty], []), ([point, empty], [0]), ([empty, point, empty], [1]), ([point, None], [0]), ([None, point, None], [1]), ([point, point], [0, 1]), ([point, line_string], [0, 1, 1, 1]), ([line_string, point], [0, 0, 0, 1]), ([line_string, linear_ring], [0, 0, 0, 1, 1, 1, 1, 1])])
def test_get_coords_index(geoms, index):
    _, actual = get_coordinates(np.array(geoms, np.object_), return_index=True)
    expected = np.array(index, dtype=np.intp)
    assert_equal(actual, expected)