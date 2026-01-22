import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(1.1, 1.1), (1.1, 1.100001), (np.int16(1), 1.000001), (np.float64(1.1), 1.1), (np.uint32(5), 5)])
def test_assert_almost_equal_numbers(a, b):
    _assert_almost_equal_both(a, b)