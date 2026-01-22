from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_pivot_no_values(self):
    idx = pd.DatetimeIndex(['2011-01-01', '2011-02-01', '2011-01-02', '2011-01-01', '2011-01-02'])
    df = DataFrame({'A': [1, 2, 3, 4, 5]}, index=idx)
    res = df.pivot_table(index=df.index.month, columns=df.index.day)
    exp_columns = MultiIndex.from_tuples([('A', 1), ('A', 2)])
    exp_columns = exp_columns.set_levels(exp_columns.levels[1].astype(np.int32), level=1)
    exp = DataFrame([[2.5, 4.0], [2.0, np.nan]], index=Index([1, 2], dtype=np.int32), columns=exp_columns)
    tm.assert_frame_equal(res, exp)
    df = DataFrame({'A': [1, 2, 3, 4, 5], 'dt': date_range('2011-01-01', freq='D', periods=5)}, index=idx)
    res = df.pivot_table(index=df.index.month, columns=Grouper(key='dt', freq='ME'))
    exp_columns = MultiIndex.from_arrays([['A'], pd.DatetimeIndex(['2011-01-31'], dtype='M8[ns]')], names=[None, 'dt'])
    exp = DataFrame([3.25, 2.0], index=Index([1, 2], dtype=np.int32), columns=exp_columns)
    tm.assert_frame_equal(res, exp)
    res = df.pivot_table(index=Grouper(freq='YE'), columns=Grouper(key='dt', freq='ME'))
    exp = DataFrame([3.0], index=pd.DatetimeIndex(['2011-12-31'], freq='YE'), columns=exp_columns)
    tm.assert_frame_equal(res, exp)