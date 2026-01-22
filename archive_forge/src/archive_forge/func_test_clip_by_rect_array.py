import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('geometry', all_types)
def test_clip_by_rect_array(geometry):
    actual = shapely.clip_by_rect([geometry, geometry], 0.0, 0.0, 1.0, 1.0)
    assert actual.shape == (2,)
    assert actual[0] is None or isinstance(actual[0], Geometry)