import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('a', all_types)
@pytest.mark.parametrize('func', SET_OPERATIONS)
def test_set_operation_array(a, func):
    if func is shapely.difference and a.geom_type == 'GeometryCollection' and (shapely.get_num_geometries(a) == 2) and (shapely.geos_version == (3, 9, 5)):
        pytest.xfail('GEOS 3.9.5 crashes with mixed collection')
    actual = func(a, point)
    assert isinstance(actual, Geometry)
    actual = func([a, a], point)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)