import numpy as np
import pytest
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
def test_insert_item_cache(self, using_array_manager, using_copy_on_write):
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)))
    ser = df[0]
    if using_array_manager:
        expected_warning = None
    else:
        expected_warning = PerformanceWarning
    with tm.assert_produces_warning(expected_warning):
        for n in range(100):
            df[n + 3] = df[1] * n
    if using_copy_on_write:
        ser.iloc[0] = 99
        assert df.iloc[0, 0] == df[0][0]
        assert df.iloc[0, 0] != 99
    else:
        ser.values[0] = 99
        assert df.iloc[0, 0] == df[0][0]
        assert df.iloc[0, 0] == 99