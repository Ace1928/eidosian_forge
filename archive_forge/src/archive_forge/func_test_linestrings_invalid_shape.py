import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('shape', [(2, 1, 2), (1, 1, 2), (1, 2)])
def test_linestrings_invalid_shape(shape):
    with pytest.raises(shapely.GEOSException):
        shapely.linestrings(np.ones(shape))