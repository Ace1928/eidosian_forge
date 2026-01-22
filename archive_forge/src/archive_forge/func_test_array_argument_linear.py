import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
def test_array_argument_linear():
    line = LineString([(0, 0), (0, 1), (1, 1)])
    distances = np.array([0, 0.5, 1])
    result = line.line_interpolate_point(distances)
    assert isinstance(result, np.ndarray)
    expected = np.array([line.line_interpolate_point(d) for d in distances], dtype=object)
    assert_geometries_equal(result, expected)
    points = shapely.points([(0, 0), (0.5, 0.5), (1, 1)])
    result = line.line_locate_point(points)
    assert isinstance(result, np.ndarray)
    expected = np.array([line.line_locate_point(p) for p in points], dtype='float64')
    np.testing.assert_array_equal(result, expected)