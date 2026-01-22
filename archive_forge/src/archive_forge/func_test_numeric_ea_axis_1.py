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
@pytest.mark.parametrize('method', ['all', 'any', 'count', 'idxmax', 'idxmin', 'kurt', 'kurtosis', 'max', 'mean', 'median', 'min', 'nunique', 'prod', 'product', 'sem', 'skew', 'std', 'sum', 'var'])
@pytest.mark.parametrize('min_count', [0, 2])
def test_numeric_ea_axis_1(method, skipna, min_count, any_numeric_ea_dtype):
    df = DataFrame({'a': Series([0, 1, 2, 3], dtype=any_numeric_ea_dtype), 'b': Series([0, 1, pd.NA, 3], dtype=any_numeric_ea_dtype)})
    expected_df = DataFrame({'a': [0.0, 1.0, 2.0, 3.0], 'b': [0.0, 1.0, np.nan, 3.0]})
    if method in ('count', 'nunique'):
        expected_dtype = 'int64'
    elif method in ('all', 'any'):
        expected_dtype = 'boolean'
    elif method in ('kurt', 'kurtosis', 'mean', 'median', 'sem', 'skew', 'std', 'var') and (not any_numeric_ea_dtype.startswith('Float')):
        expected_dtype = 'Float64'
    else:
        expected_dtype = any_numeric_ea_dtype
    kwargs = {}
    if method not in ('count', 'nunique', 'quantile'):
        kwargs['skipna'] = skipna
    if method in ('prod', 'product', 'sum'):
        kwargs['min_count'] = min_count
    warn = None
    msg = None
    if not skipna and method in ('idxmax', 'idxmin'):
        warn = FutureWarning
        msg = f'The behavior of DataFrame.{method} with all-NA values'
    with tm.assert_produces_warning(warn, match=msg):
        result = getattr(df, method)(axis=1, **kwargs)
    with tm.assert_produces_warning(warn, match=msg):
        expected = getattr(expected_df, method)(axis=1, **kwargs)
    if method not in ('idxmax', 'idxmin'):
        expected = expected.astype(expected_dtype)
    tm.assert_series_equal(result, expected)