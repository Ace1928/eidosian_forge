import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('func, related_func', REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_empty(func, related_func):
    assert func(np.empty((0,), dtype=object)) == empty
    arr_empty_2D = np.empty((0, 2), dtype=object)
    assert func(arr_empty_2D) == empty
    assert func(arr_empty_2D, axis=0).tolist() == [empty] * 2
    assert func(arr_empty_2D, axis=1).tolist() == []