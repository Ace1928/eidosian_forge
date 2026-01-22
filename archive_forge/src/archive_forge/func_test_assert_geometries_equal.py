import numpy as np
import pytest
import shapely
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('use_array', ['none', 'left', 'right', 'both'])
@pytest.mark.parametrize('geom', all_types + EMPTY_GEOMS)
def test_assert_geometries_equal(geom, use_array):
    assert_geometries_equal(*make_array(geom, geom, use_array))