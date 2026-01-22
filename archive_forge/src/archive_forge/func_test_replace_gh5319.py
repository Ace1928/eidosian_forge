import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_gh5319(self):
    ser = pd.Series([0, np.nan, 2, 3, 4])
    expected = ser.ffill()
    msg = "Series.replace without 'value' and with non-dict-like 'to_replace' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ser.replace([np.nan])
    tm.assert_series_equal(result, expected)
    ser = pd.Series([0, np.nan, 2, 3, 4])
    expected = ser.ffill()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ser.replace(np.nan)
    tm.assert_series_equal(result, expected)