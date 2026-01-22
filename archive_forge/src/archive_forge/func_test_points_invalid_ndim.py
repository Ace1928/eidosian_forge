import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_points_invalid_ndim():
    with pytest.raises(ValueError, match='dimension should be 2 or 3, got 4'):
        shapely.points([0, 1, 2, 3])
    with pytest.raises(ValueError, match='dimension should be 2 or 3, got 1'):
        shapely.points([0])