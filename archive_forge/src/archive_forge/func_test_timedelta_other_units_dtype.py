from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['timedelta64[D]', 'timedelta64[h]', 'timedelta64[m]', 'timedelta64[s]', 'timedelta64[ms]', 'timedelta64[us]', 'timedelta64[ns]'])
def test_timedelta_other_units_dtype(self, dtype):
    idx = TimedeltaIndex(['1 days', 'NaT', '2 days'])
    values = idx.values.astype(dtype)
    exp = np.array([False, True, False])
    tm.assert_numpy_array_equal(isna(values), exp)
    tm.assert_numpy_array_equal(notna(values), ~exp)
    exp = Series([False, True, False])
    s = Series(values)
    tm.assert_series_equal(isna(s), exp)
    tm.assert_series_equal(notna(s), ~exp)
    s = Series(values, dtype=object)
    tm.assert_series_equal(isna(s), exp)
    tm.assert_series_equal(notna(s), ~exp)