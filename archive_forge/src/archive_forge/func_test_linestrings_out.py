import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
@pytest.mark.parametrize('indices,expected', [([0, 0, 0, 1, 1, 1], [line_string, line_string, empty_point, None]), ([0, 0, 0, 3, 3, 3], [line_string, None, empty_point, line_string]), ([2, 2, 2, 3, 3, 3], [None, None, line_string, line_string])])
def test_linestrings_out(indices, expected):
    out = np.empty(4, dtype=object)
    out[2] = empty_point
    actual = shapely.linestrings([(0, 0), (1, 0), (1, 1), (0, 0), (1, 0), (1, 1)], indices=indices, out=out)
    assert_geometries_equal(out, expected)
    assert actual is out