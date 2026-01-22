import numpy as np
import pytest
from shapely import Point
from shapely.coords import CoordinateSequence
from shapely.errors import DimensionError
def test_from_numpy_xy():
    p = Point(np.array([1.0]), np.array([2.0]))
    assert p.coords[:] == [(1.0, 2.0)]
    p = Point(np.array([1.0]), np.array([2.0]), np.array([3.0]))
    assert p.coords[:] == [(1.0, 2.0, 3.0)]