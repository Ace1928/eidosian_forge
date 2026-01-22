import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_datetime64(self):
    ser = pd.Series(pd.date_range('20130101', periods=5))
    expected = ser.copy()
    expected.loc[2] = pd.Timestamp('20120101')
    result = ser.replace({pd.Timestamp('20130103'): pd.Timestamp('20120101')})
    tm.assert_series_equal(result, expected)
    result = ser.replace(pd.Timestamp('20130103'), pd.Timestamp('20120101'))
    tm.assert_series_equal(result, expected)