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
@pytest.mark.parametrize('min_count', [0, 10])
def test_groupby_sum_mincount_boolean(min_count):
    b = True
    a = False
    na = np.nan
    dfg = pd.array([b, b, na, na, a, a, b], dtype='boolean')
    df = DataFrame({'A': [1, 1, 2, 2, 3, 3, 1], 'B': dfg})
    result = df.groupby('A').sum(min_count=min_count)
    if min_count == 0:
        expected = DataFrame({'B': pd.array([3, 0, 0], dtype='Int64')}, index=pd.Index([1, 2, 3], name='A'))
        tm.assert_frame_equal(result, expected)
    else:
        expected = DataFrame({'B': pd.array([pd.NA] * 3, dtype='Int64')}, index=pd.Index([1, 2, 3], name='A'))
        tm.assert_frame_equal(result, expected)