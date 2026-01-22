import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version[:2] != (3, 12), reason='GEOS != 3.12')
def test_points_nan_3D_all_nan_becomes_empty():
    actual = shapely.points(np.nan, np.nan, np.nan)
    assert actual.wkt == 'POINT Z EMPTY'