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
def test_min_max_dt64_api_consistency_with_NaT(self):
    df = DataFrame({'x': to_datetime([])})
    expected_dt_series = Series(to_datetime([]))
    assert (df.min(axis=0).x is pd.NaT) == (expected_dt_series.min() is pd.NaT)
    assert (df.max(axis=0).x is pd.NaT) == (expected_dt_series.max() is pd.NaT)
    tm.assert_series_equal(df.min(axis=1), expected_dt_series)
    tm.assert_series_equal(df.max(axis=1), expected_dt_series)