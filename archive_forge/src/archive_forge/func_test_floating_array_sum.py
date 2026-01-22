import numpy as np
import pytest
from pandas.compat import IS64
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.parametrize('min_count', [0, 4])
def test_floating_array_sum(skipna, min_count, dtype):
    arr = pd.array([1, 2, 3, None], dtype=dtype)
    result = arr.sum(skipna=skipna, min_count=min_count)
    if skipna and min_count == 0:
        assert result == 6.0
    else:
        assert result is pd.NA