import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
@pytest.mark.parametrize('ser, exp', [([1, 2, 3], [1, True, 3]), (['x', 2, 3], ['x', True, 3])])
def test_replace_no_cast(self, ser, exp):
    series = pd.Series(ser)
    result = series.replace(2, True)
    expected = pd.Series(exp)
    tm.assert_series_equal(result, expected)