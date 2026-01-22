import numpy as np
import pytest
import shapely
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('use_array', ['none', 'left', 'right', 'both'])
def test_assert_none_not_equal(use_array):
    with pytest.raises(AssertionError):
        assert_geometries_equal(*make_array(None, None, use_array), equal_none=False)