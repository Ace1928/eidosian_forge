import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
@pytest.mark.parametrize('op', ['distance', 'hausdorff_distance'])
def test_array_argument_float(op):
    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    points = shapely.points([(0, 0), (0.5, 0.5), (1, 1)])
    result = getattr(polygon, op)(points)
    assert isinstance(result, np.ndarray)
    expected = np.array([getattr(polygon, op)(p) for p in points], dtype='float64')
    np.testing.assert_array_equal(result, expected)