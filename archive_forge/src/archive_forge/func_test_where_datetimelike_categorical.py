import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_where_datetimelike_categorical(tz_naive_fixture):
    tz = tz_naive_fixture
    dr = date_range('2001-01-01', periods=3, tz=tz)._with_freq(None)
    lvals = pd.DatetimeIndex([dr[0], dr[1], pd.NaT])
    rvals = pd.Categorical([dr[0], pd.NaT, dr[2]])
    mask = np.array([True, True, False])
    res = lvals.where(mask, rvals)
    tm.assert_index_equal(res, dr)
    res = lvals._data._where(mask, rvals)
    tm.assert_datetime_array_equal(res, dr._data)
    res = Series(lvals).where(mask, rvals)
    tm.assert_series_equal(res, Series(dr))
    res = pd.DataFrame(lvals).where(mask[:, None], pd.DataFrame(rvals))
    tm.assert_frame_equal(res, pd.DataFrame(dr))