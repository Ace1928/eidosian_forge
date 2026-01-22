from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_sum_timedelta64_skipna_false(using_array_manager, request):
    if using_array_manager:
        mark = pytest.mark.xfail(reason='Incorrect type inference on NaT in reduction result')
        request.applymarker(mark)
    arr = np.arange(8).astype(np.int64).view('m8[s]').reshape(4, 2)
    arr[-1, -1] = 'Nat'
    df = DataFrame(arr)
    assert (df.dtypes == arr.dtype).all()
    result = df.sum(skipna=False)
    expected = Series([pd.Timedelta(seconds=12), pd.NaT], dtype='m8[s]')
    tm.assert_series_equal(result, expected)
    result = df.sum(axis=0, skipna=False)
    tm.assert_series_equal(result, expected)
    result = df.sum(axis=1, skipna=False)
    expected = Series([pd.Timedelta(seconds=1), pd.Timedelta(seconds=5), pd.Timedelta(seconds=9), pd.NaT], dtype='m8[s]')
    tm.assert_series_equal(result, expected)