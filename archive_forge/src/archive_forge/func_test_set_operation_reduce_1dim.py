import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('n', range(1, 5))
@pytest.mark.parametrize('func, related_func', REDUCE_SET_OPERATIONS)
def test_set_operation_reduce_1dim(n, func, related_func):
    actual = func(reduce_test_data[:n])
    expected = reduce_test_data[0]
    for i in range(1, n):
        expected = related_func(expected, reduce_test_data[i])
    assert shapely.equals(actual, expected)