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
@pytest.mark.parametrize('dtype', ['float64', 'float32', 'int64', 'int32', 'int16', 'int8'])
def test_with_na_groups(dtype):
    index = Index(np.arange(10))
    values = Series(np.ones(10), index, dtype=dtype)
    labels = Series([np.nan, 'foo', 'bar', 'bar', np.nan, np.nan, 'bar', 'bar', np.nan, 'foo'], index=index)
    grouped = values.groupby(labels)
    agged = grouped.agg(len)
    expected = Series([4, 2], index=['bar', 'foo'])
    tm.assert_series_equal(agged, expected, check_dtype=False)

    def f(x):
        return float(len(x))
    agged = grouped.agg(f)
    expected = Series([4.0, 2.0], index=['bar', 'foo'])
    tm.assert_series_equal(agged, expected)