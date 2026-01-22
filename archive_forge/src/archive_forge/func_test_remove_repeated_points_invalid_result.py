import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason='GEOS < 3.12')
@pytest.mark.parametrize('geom, tolerance', [[Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]), 2]])
def test_remove_repeated_points_invalid_result(geom, tolerance):
    with pytest.raises(shapely.GEOSException, match='Invalid number of points'):
        shapely.remove_repeated_points(geom, tolerance)