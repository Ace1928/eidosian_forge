from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
@pytest.mark.parametrize('timezone', ['UTC', 'US/Pacific', 'GMT'])
def test_interval_index_subtype(timezone, inclusive_endpoints_fixture):
    dates = date_range('2022', periods=3, tz=timezone)
    dtype = f'interval[datetime64[ns, {timezone}], {inclusive_endpoints_fixture}]'
    result = IntervalIndex.from_arrays(['2022-01-01', '2022-01-02'], ['2022-01-02', '2022-01-03'], closed=inclusive_endpoints_fixture, dtype=dtype)
    expected = IntervalIndex.from_arrays(dates[:-1], dates[1:], closed=inclusive_endpoints_fixture)
    tm.assert_index_equal(result, expected)