from datetime import timedelta
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.timedeltas import timedelta_range
@pytest.mark.parametrize('start, end, freq, resample_freq', [('8h', '21h59min50s', '10s', '3h'), ('3h', '22h', '1h', '5h'), ('527D', '5006D', '3D', '10D'), ('1D', '10D', '1D', '2D'), ('8h', '21h59min50s', '10s', '2h'), ('0h', '21h59min50s', '10s', '3h'), ('10D', '85D', 'D', '2D')])
def test_resample_timedelta_edge_case(start, end, freq, resample_freq):
    idx = timedelta_range(start=start, end=end, freq=freq)
    s = Series(np.arange(len(idx)), index=idx)
    result = s.resample(resample_freq).min()
    expected_index = timedelta_range(freq=resample_freq, start=start, end=end)
    tm.assert_index_equal(result.index, expected_index)
    assert result.index.freq == expected_index.freq
    assert not np.isnan(result.iloc[-1])