import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('shape', [(2, 1, 2), (1, 1, 2), (1, 2), (2, 2, 2), (1, 2, 2), (2, 2)])
def test_linearrings_invalid_shape(shape):
    coords = np.ones(shape)
    with pytest.raises(ValueError):
        shapely.linearrings(coords)
    coords[..., 1] += 1
    with pytest.raises(ValueError):
        shapely.linearrings(coords)