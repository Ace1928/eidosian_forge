import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_with_no_overflowerror(self):
    s = pd.Series([0, 1, 2, 3, 4])
    result = s.replace([3], ['100000000000000000000'])
    expected = pd.Series([0, 1, 2, '100000000000000000000', 4])
    tm.assert_series_equal(result, expected)
    s = pd.Series([0, '100000000000000000000', '100000000000000000001'])
    result = s.replace(['100000000000000000000'], [1])
    expected = pd.Series([0, 1, '100000000000000000001'])
    tm.assert_series_equal(result, expected)