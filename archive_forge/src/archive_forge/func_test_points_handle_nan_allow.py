import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 12, 0), reason='GEOS < 3.12')
@pytest.mark.parametrize('coords,expected_wkt', [pytest.param([np.nan, np.nan], 'POINT (NaN NaN)', marks=pytest.mark.skipif(shapely.geos_version < (3, 13, 0), reason='GEOS < 3.13')), pytest.param([np.nan, np.nan, np.nan], 'POINT Z (NaN NaN NaN)', marks=pytest.mark.skipif(shapely.geos_version < (3, 13, 0), reason='GEOS < 3.13')), ([1, np.nan], 'POINT (1 NaN)'), ([np.nan, 1], 'POINT (NaN 1)'), ([np.nan, 1, np.nan], 'POINT Z (NaN 1 NaN)'), ([np.nan, np.nan, 1], 'POINT Z (NaN NaN 1)')])
def test_points_handle_nan_allow(coords, expected_wkt):
    actual = shapely.points(coords)
    assert actual.wkt == expected_wkt