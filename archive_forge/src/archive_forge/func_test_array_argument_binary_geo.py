import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
@pytest.mark.parametrize('op', ['difference', 'intersection', 'symmetric_difference', 'union'])
def test_array_argument_binary_geo(op):
    box = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    polygons = shapely.buffer(shapely.points([(0, 0), (0.5, 0.5), (1, 1)]), 0.5)
    result = getattr(box, op)(polygons)
    assert isinstance(result, np.ndarray)
    expected = np.array([getattr(box, op)(g) for g in polygons], dtype=object)
    assert_geometries_equal(result, expected)