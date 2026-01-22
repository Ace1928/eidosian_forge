import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('geometry', all_types)
@pytest.mark.parametrize('reference', all_types)
def test_snap_array(geometry, reference):
    actual = shapely.snap([geometry, geometry], [reference, reference], tolerance=1.0)
    assert actual.shape == (2,)
    assert isinstance(actual[0], Geometry)