import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('indices,expected', [([0, 1], [poly, poly, empty_point, None]), ([0, 3], [poly, None, empty_point, poly]), ([2, 3], [None, None, poly, poly])])
def test_polygons_out(indices, expected):
    out = np.empty(4, dtype=object)
    out[2] = empty_point
    actual = shapely.polygons([linear_ring, linear_ring], indices=indices, out=out)
    assert_geometries_equal(out, expected)
    assert actual is out