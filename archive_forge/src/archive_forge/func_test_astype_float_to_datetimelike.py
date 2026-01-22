import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['M8[ns]', 'm8[ns]'])
def test_astype_float_to_datetimelike(self, dtype):
    idx = Index([0, 1.1, 2], dtype=np.float64)
    result = idx.astype(dtype)
    if dtype[0] == 'M':
        expected = to_datetime(idx.values)
    else:
        expected = to_timedelta(idx.values)
    tm.assert_index_equal(result, expected)
    result = idx.to_series().set_axis(range(3)).astype(dtype)
    expected = expected.to_series().set_axis(range(3))
    tm.assert_series_equal(result, expected)