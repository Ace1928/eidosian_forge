import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
def test_set_coords_0dim():
    actual = set_coordinates(point, [[1, 1]])
    assert isinstance(actual, shapely.Geometry)
    actual = set_coordinates(np.asarray(point), [[1, 1]])
    assert isinstance(actual, np.ndarray)
    assert actual.ndim == 0