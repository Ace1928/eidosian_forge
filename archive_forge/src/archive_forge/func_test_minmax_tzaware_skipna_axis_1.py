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
@pytest.mark.parametrize('method', ['min', 'max'])
def test_minmax_tzaware_skipna_axis_1(self, method, skipna):
    val = to_datetime('1900-01-01', utc=True)
    df = DataFrame({'a': Series([pd.NaT, pd.NaT, val]), 'b': Series([pd.NaT, val, val])})
    op = getattr(df, method)
    result = op(axis=1, skipna=skipna)
    if skipna:
        expected = Series([pd.NaT, val, val])
    else:
        expected = Series([pd.NaT, pd.NaT, val])
    tm.assert_series_equal(result, expected)