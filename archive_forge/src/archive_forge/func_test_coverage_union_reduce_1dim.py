import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason='GEOS < 3.8')
@pytest.mark.parametrize('n', range(1, 4))
def test_coverage_union_reduce_1dim(n):
    """
    This is tested seperately from other set operations as it differs in two ways:
      1. It expects only non-overlapping polygons
      2. It expects GEOS 3.8.0+
    """
    test_data = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1), shapely.box(2, 0, 3, 1)]
    actual = shapely.coverage_union_all(test_data[:n])
    expected = test_data[0]
    for i in range(1, n):
        expected = shapely.coverage_union(expected, test_data[i])
    assert_geometries_equal(actual, expected, normalize=True)