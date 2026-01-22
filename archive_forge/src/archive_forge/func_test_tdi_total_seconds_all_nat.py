import numpy as np
import pytest
from pandas._libs.tslibs.offsets import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_tdi_total_seconds_all_nat(self):
    ser = Series([np.nan, np.nan], dtype='timedelta64[ns]')
    result = ser.dt.total_seconds()
    expected = Series([np.nan, np.nan])
    tm.assert_series_equal(result, expected)