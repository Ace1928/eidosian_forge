import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(0.001, 0), (1, 0)])
def test_assert_not_almost_equal_numbers_with_zeros(a, b):
    _assert_not_almost_equal_both(a, b)