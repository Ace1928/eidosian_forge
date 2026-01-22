import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(np.inf, np.inf), (np.inf, float('inf')), (np.array([np.inf, np.nan, -np.inf]), np.array([np.inf, np.nan, -np.inf]))])
def test_assert_almost_equal_inf(a, b):
    _assert_almost_equal_both(a, b)