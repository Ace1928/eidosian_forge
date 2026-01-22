import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('func', [shapely.points, shapely.linestrings, shapely.linearrings])
@pytest.mark.parametrize('indices', [[point], ' hello', [0, 1], [-1]])
def test_invalid_indices_simple(func, indices):
    with pytest.raises((TypeError, ValueError)):
        func([[0.2, 0.3]], indices=indices)