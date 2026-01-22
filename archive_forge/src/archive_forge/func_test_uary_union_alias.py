import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 9, 0), reason='GEOS < 3.9')
def test_uary_union_alias():
    geoms = [shapely.box(0.1, 0.1, 5, 5), shapely.box(0, 0.2, 5.1, 10)]
    actual = shapely.unary_union(geoms, grid_size=1)
    expected = shapely.union_all(geoms, grid_size=1)
    assert shapely.equals(actual, expected)