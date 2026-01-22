import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason='GEOS < 3.11')
@pytest.mark.parametrize('geom, tolerance', [('Not a geometry', 1), (1, 1)])
def test_remove_repeated_points_invalid_type(geom, tolerance):
    with pytest.raises(TypeError, match='One of the arguments is of incorrect type'):
        shapely.remove_repeated_points(geom, tolerance)