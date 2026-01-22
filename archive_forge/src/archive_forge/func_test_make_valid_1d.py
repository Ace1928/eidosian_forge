import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason='GEOS < 3.8')
@pytest.mark.parametrize('geom,expected', [(all_types, all_types), ([Polygon([(0, 0), (2, 2), (0, 2), (0, 0)]), Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])], [Polygon([(0, 0), (2, 2), (0, 2), (0, 0)]), MultiPolygon([Polygon([(1, 1), (0, 0), (0, 2), (1, 1)]), Polygon([(1, 1), (2, 2), (2, 0), (1, 1)])])]), ([point, None, empty], [point, None, empty])])
def test_make_valid_1d(geom, expected):
    actual = shapely.make_valid(geom)
    assert np.all(shapely.normalize(actual) == shapely.normalize(expected))