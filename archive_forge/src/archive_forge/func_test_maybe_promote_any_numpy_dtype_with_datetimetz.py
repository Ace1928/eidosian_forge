import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs import NaT
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
import pandas as pd
@pytest.mark.parametrize('fill_value', [pd.Timestamp(2023, 1, 1), np.datetime64('2023-01-01'), datetime.datetime(2023, 1, 1), datetime.date(2023, 1, 1)], ids=['pd.Timestamp', 'np.datetime64', 'datetime.datetime', 'datetime.date'])
def test_maybe_promote_any_numpy_dtype_with_datetimetz(any_numpy_dtype, tz_aware_fixture, fill_value):
    dtype = np.dtype(any_numpy_dtype)
    fill_dtype = DatetimeTZDtype(tz=tz_aware_fixture)
    fill_value = pd.Series([fill_value], dtype=fill_dtype)[0]
    expected_dtype = np.dtype(object)
    exp_val_for_scalar = fill_value
    _check_promote(dtype, fill_value, expected_dtype, exp_val_for_scalar)