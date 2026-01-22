import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interp_time_inplace_axis(self):
    periods = 5
    idx = date_range(start='2014-01-01', periods=periods)
    data = np.random.default_rng(2).random((periods, periods))
    data[data < 0.5] = np.nan
    expected = DataFrame(index=idx, columns=idx, data=data)
    result = expected.interpolate(axis=0, method='time')
    return_value = expected.interpolate(axis=0, method='time', inplace=True)
    assert return_value is None
    tm.assert_frame_equal(result, expected)