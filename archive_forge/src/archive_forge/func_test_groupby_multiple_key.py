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
def test_groupby_multiple_key():
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    grouped = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    agged = grouped.sum()
    tm.assert_almost_equal(df.values, agged.values)
    depr_msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        grouped = df.T.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day], axis=1)
    agged = grouped.agg(lambda x: x.sum())
    tm.assert_index_equal(agged.index, df.columns)
    tm.assert_almost_equal(df.T.values, agged.values)
    agged = grouped.agg(lambda x: x.sum())
    tm.assert_almost_equal(df.T.values, agged.values)