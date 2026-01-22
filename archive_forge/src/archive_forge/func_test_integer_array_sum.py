import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.parametrize('min_count', [0, 4])
def test_integer_array_sum(skipna, min_count, any_int_ea_dtype):
    dtype = any_int_ea_dtype
    arr = pd.array([1, 2, 3, None], dtype=dtype)
    result = arr.sum(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        assert result == 6
    else:
        assert result is pd.NA