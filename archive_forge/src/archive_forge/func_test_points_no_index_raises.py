import numpy as np
import pytest
import shapely
from shapely import LinearRing, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import empty_point, line_string, linear_ring, point, polygon
def test_points_no_index_raises():
    with pytest.raises(ValueError):
        shapely.points(np.array([[2, 3], [2, 3]], dtype=float), indices=np.array([0, 2], dtype=np.intp))