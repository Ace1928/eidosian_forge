import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 9, 0), reason='GEOS < 3.9')
@pytest.mark.parametrize('func, related_func', REDUCE_SET_OPERATIONS_PREC)
def test_set_operation_prec_reduce_nonscalar_grid_size(func, related_func):
    with pytest.raises(ValueError, match='grid_size parameter only accepts scalar values'):
        func([point, point], grid_size=[1])