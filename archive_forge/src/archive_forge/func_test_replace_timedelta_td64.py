import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_timedelta_td64(self):
    tdi = pd.timedelta_range(0, periods=5)
    ser = pd.Series(tdi)
    result = ser.replace({ser[1]: ser[3]})
    expected = pd.Series([ser[0], ser[3], ser[2], ser[3], ser[4]])
    tm.assert_series_equal(result, expected)