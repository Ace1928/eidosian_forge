import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('geometry', all_types)
@pytest.mark.parametrize('func', CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_array(geometry, func):
    if func is shapely.offset_curve and shapely.get_type_id(geometry) not in [1, 2] and (shapely.geos_version < (3, 11, 0)):
        with pytest.raises(GEOSException, match='only accept linestrings'):
            func([geometry, geometry], 0.0)
        return
    with ignore_invalid(func is shapely.voronoi_polygons and shapely.get_type_id(geometry) == 0 and (shapely.geos_version < (3, 12, 0))):
        actual = func([geometry, geometry], 0.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)