import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_with_single_list(self):
    ser = pd.Series([0, 1, 2, 3, 4])
    msg2 = "Series.replace without 'value' and with non-dict-like 'to_replace' is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg2):
        result = ser.replace([1, 2, 3])
    tm.assert_series_equal(result, pd.Series([0, 0, 0, 0, 4]))
    s = ser.copy()
    with tm.assert_produces_warning(FutureWarning, match=msg2):
        return_value = s.replace([1, 2, 3], inplace=True)
    assert return_value is None
    tm.assert_series_equal(s, pd.Series([0, 0, 0, 0, 4]))
    s = ser.copy()
    msg = 'Invalid fill method\\. Expecting pad \\(ffill\\) or backfill \\(bfill\\)\\. Got crash_cymbal'
    msg3 = "The 'method' keyword in Series.replace is deprecated"
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=msg3):
            return_value = s.replace([1, 2, 3], inplace=True, method='crash_cymbal')
        assert return_value is None
    tm.assert_series_equal(s, ser)