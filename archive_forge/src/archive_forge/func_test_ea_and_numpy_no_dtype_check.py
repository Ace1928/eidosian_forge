import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [None, object])
@pytest.mark.parametrize('check_exact', [True, False])
@pytest.mark.parametrize('val', [3, 3.5])
def test_ea_and_numpy_no_dtype_check(val, check_exact, dtype):
    left = Series([1, 2, val], dtype=dtype)
    right = Series(pd.array([1, 2, val]))
    tm.assert_series_equal(left, right, check_dtype=False, check_exact=check_exact)