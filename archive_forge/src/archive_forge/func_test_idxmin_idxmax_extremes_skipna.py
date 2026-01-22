import builtins
import datetime as dt
from string import ascii_lowercase
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
@pytest.mark.parametrize('how', ['idxmin', 'idxmax'])
def test_idxmin_idxmax_extremes_skipna(skipna, how, float_numpy_dtype):
    min_value = np.finfo(float_numpy_dtype).min
    max_value = np.finfo(float_numpy_dtype).max
    df = DataFrame({'a': Series(np.repeat(range(1, 6), repeats=2), dtype='intp'), 'b': Series([np.nan, min_value, np.nan, max_value, min_value, np.nan, max_value, np.nan, np.nan, np.nan], dtype=float_numpy_dtype)})
    gb = df.groupby('a')
    warn = None if skipna else FutureWarning
    msg = f'The behavior of DataFrameGroupBy.{how} with all-NA values'
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(gb, how)(skipna=skipna)
    if skipna:
        values = [1, 3, 4, 6, np.nan]
    else:
        values = np.nan
    expected = DataFrame({'b': values}, index=pd.Index(range(1, 6), name='a', dtype='intp'))
    tm.assert_frame_equal(result, expected)