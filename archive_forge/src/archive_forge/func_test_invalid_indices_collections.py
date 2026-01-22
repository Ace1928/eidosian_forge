import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('func', [shapely.polygons, shapely.multipoints, shapely.multilinestrings, shapely.multipolygons, shapely.geometrycollections])
@pytest.mark.parametrize('indices', [np.array([point]), ' hello', [0, 1], [-1]])
def test_invalid_indices_collections(func, indices):
    with pytest.raises((TypeError, ValueError)):
        func([point], indices=indices)