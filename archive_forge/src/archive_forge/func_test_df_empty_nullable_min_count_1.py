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
@pytest.mark.parametrize('opname, dtype, exp_dtype', [('sum', 'Int8', 'Int32' if is_windows_or_is32 else 'Int64'), ('prod', 'Int8', 'Int32' if is_windows_or_is32 else 'Int64'), ('sum', 'Int64', 'Int64'), ('prod', 'Int64', 'Int64'), ('sum', 'UInt8', 'UInt32' if is_windows_or_is32 else 'UInt64'), ('prod', 'UInt8', 'UInt32' if is_windows_or_is32 else 'UInt64'), ('sum', 'UInt64', 'UInt64'), ('prod', 'UInt64', 'UInt64'), ('sum', 'Float32', 'Float32'), ('prod', 'Float32', 'Float32'), ('sum', 'Float64', 'Float64')])
def test_df_empty_nullable_min_count_1(self, opname, dtype, exp_dtype):
    df = DataFrame({0: [], 1: []}, dtype=dtype)
    result = getattr(df, opname)(min_count=1)
    expected = Series([pd.NA, pd.NA], dtype=exp_dtype)
    tm.assert_series_equal(result, expected)