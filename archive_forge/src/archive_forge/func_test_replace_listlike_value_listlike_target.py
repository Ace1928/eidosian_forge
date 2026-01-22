import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_listlike_value_listlike_target(self, datetime_series):
    ser = pd.Series(datetime_series.index)
    tm.assert_series_equal(ser.replace(np.nan, 0), ser.fillna(0))
    msg = 'Replacement lists must match in length\\. Expecting 3 got 2'
    with pytest.raises(ValueError, match=msg):
        ser.replace([1, 2, 3], [np.nan, 0])
    result = ser.replace([1, 2], [np.nan, 0])
    tm.assert_series_equal(result, ser)
    ser = pd.Series([0, 1, 2, 3, 4])
    result = ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    tm.assert_series_equal(result, pd.Series([4, 3, 2, 1, 0]))