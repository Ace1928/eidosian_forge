import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@td.skip_if_no('pyarrow')
def test_arrow_datetime_axis():
    expected = Series(np.arange(5, dtype=np.float64), index=Index(date_range('2020-01-01', periods=5), dtype='timestamp[ns][pyarrow]'))
    result = expected.rolling('1D').sum()
    tm.assert_series_equal(result, expected)