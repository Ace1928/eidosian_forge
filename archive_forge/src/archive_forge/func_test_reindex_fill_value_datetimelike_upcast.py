import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_not_yet_implemented
@pytest.mark.parametrize('dtype', ['datetime64[ns]', 'timedelta64[ns]'])
@pytest.mark.parametrize('fill_value', ['string', 0, Timedelta(0)])
def test_reindex_fill_value_datetimelike_upcast(dtype, fill_value, using_array_manager):
    if dtype == 'timedelta64[ns]' and fill_value == Timedelta(0):
        fill_value = Timestamp(0)
    ser = Series([NaT], dtype=dtype)
    result = ser.reindex([0, 1], fill_value=fill_value)
    expected = Series([NaT, fill_value], index=[0, 1], dtype=object)
    tm.assert_series_equal(result, expected)