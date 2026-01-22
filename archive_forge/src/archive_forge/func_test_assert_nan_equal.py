import numpy as np
import pytest
import shapely
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('use_array', ['none', 'left', 'right', 'both'])
def test_assert_nan_equal(use_array):
    assert_geometries_equal(*make_array(line_string_nan, line_string_nan, use_array))