import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_noop_doesnt_downcast(self):
    ser = pd.Series([None, None, pd.Timestamp('2021-12-16 17:31')], dtype=object)
    res = ser.replace({np.nan: None})
    tm.assert_series_equal(res, ser)
    assert res.dtype == object
    res = ser.replace(np.nan, None)
    tm.assert_series_equal(res, ser)
    assert res.dtype == object