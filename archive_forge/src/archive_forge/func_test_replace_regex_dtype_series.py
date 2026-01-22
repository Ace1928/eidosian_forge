import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't fill 1 in string")
@pytest.mark.parametrize('regex', [False, True])
def test_replace_regex_dtype_series(self, regex):
    series = pd.Series(['0'])
    expected = pd.Series([1])
    msg = 'Downcasting behavior in `replace`'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = series.replace(to_replace='0', value=1, regex=regex)
    tm.assert_series_equal(result, expected)