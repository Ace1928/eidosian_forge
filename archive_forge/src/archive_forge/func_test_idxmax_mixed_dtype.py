from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
def test_idxmax_mixed_dtype(self):
    dti = date_range('2016-01-01', periods=3)
    df = DataFrame({1: [0, 2, 1], 2: range(3)[::-1], 3: dti.copy(deep=True)})
    result = df.idxmax()
    expected = Series([1, 0, 2], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)
    result = df.idxmin()
    expected = Series([0, 2, 0], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)
    df.loc[0, 3] = pd.NaT
    result = df.idxmax()
    expected = Series([1, 0, 2], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)
    result = df.idxmin()
    expected = Series([0, 2, 1], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)
    df[4] = dti[::-1]
    df._consolidate_inplace()
    result = df.idxmax()
    expected = Series([1, 0, 2, 0], index=[1, 2, 3, 4])
    tm.assert_series_equal(result, expected)
    result = df.idxmin()
    expected = Series([0, 2, 1, 2], index=[1, 2, 3, 4])
    tm.assert_series_equal(result, expected)