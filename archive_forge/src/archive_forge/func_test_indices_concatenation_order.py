from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_indices_concatenation_order():

    def f1(x):
        y = x[x.b % 2 == 1] ** 2
        if y.empty:
            multiindex = MultiIndex(levels=[[]] * 2, codes=[[]] * 2, names=['b', 'c'])
            res = DataFrame(columns=['a'], index=multiindex)
            return res
        else:
            y = y.set_index(['b', 'c'])
            return y

    def f2(x):
        y = x[x.b % 2 == 1] ** 2
        if y.empty:
            return DataFrame()
        else:
            y = y.set_index(['b', 'c'])
            return y

    def f3(x):
        y = x[x.b % 2 == 1] ** 2
        if y.empty:
            multiindex = MultiIndex(levels=[[]] * 2, codes=[[]] * 2, names=['foo', 'bar'])
            res = DataFrame(columns=['a', 'b'], index=multiindex)
            return res
        else:
            return y
    df = DataFrame({'a': [1, 2, 2, 2], 'b': range(4), 'c': range(5, 9)})
    df2 = DataFrame({'a': [3, 2, 2, 2], 'b': range(4), 'c': range(5, 9)})
    depr_msg = 'The behavior of array concatenation with empty entries is deprecated'
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result1 = df.groupby('a').apply(f1)
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result2 = df2.groupby('a').apply(f1)
    tm.assert_frame_equal(result1, result2)
    msg = 'Cannot concat indices that do not have the same number of levels'
    with pytest.raises(AssertionError, match=msg):
        df.groupby('a').apply(f2)
    with pytest.raises(AssertionError, match=msg):
        df2.groupby('a').apply(f2)
    with pytest.raises(AssertionError, match=msg):
        df.groupby('a').apply(f3)
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            df2.groupby('a').apply(f3)