import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('as_period', [True, False])
def test_mode_extension_dtype(as_period):
    ser = Series([pd.Timestamp(1979, 4, n) for n in range(1, 5)])
    if as_period:
        ser = ser.dt.to_period('D')
    else:
        ser = ser.dt.tz_localize('US/Central')
    res = ser.mode()
    assert res.dtype == ser.dtype
    tm.assert_series_equal(res, ser)