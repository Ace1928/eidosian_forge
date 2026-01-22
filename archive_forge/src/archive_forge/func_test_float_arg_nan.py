import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('geometry', all_types)
@pytest.mark.parametrize('func', CONSTRUCTIVE_FLOAT_ARG)
def test_float_arg_nan(geometry, func):
    actual = func(geometry, float('nan'))
    assert actual is None