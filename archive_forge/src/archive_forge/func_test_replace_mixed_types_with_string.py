import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_replace_mixed_types_with_string(self):
    s = pd.Series([1, 2, 3, '4', 4, 5])
    msg = 'Downcasting behavior in `replace`'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.replace([2, '4'], np.nan)
    expected = pd.Series([1, np.nan, 3, np.nan, 4, 5])
    tm.assert_series_equal(expected, result)