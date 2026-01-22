import datetime as dt
from datetime import date
import re
import numpy as np
import pytest
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq, expected_values, freq_depr', [('2BYE-MAR', ['2016-03-31'], '2BA-MAR'), ('2BYE-JUN', ['2016-06-30'], '2BY-JUN'), ('2BME', ['2016-02-29', '2016-04-29', '2016-06-30'], '2BM'), ('2BQE', ['2016-03-31'], '2BQ'), ('1BQE-MAR', ['2016-03-31', '2016-06-30'], '1BQ-MAR')])
def test_BM_BQ_BY_deprecated(self, freq, expected_values, freq_depr):
    msg = f"'{freq_depr[1:]}' is deprecated and will be removed "
    f"in a future version, please use '{freq[1:]}' instead."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = date_range(start='2016-02-21', end='2016-08-21', freq=freq_depr)
    result = DatetimeIndex(data=expected_values, dtype='datetime64[ns]', freq=freq)
    tm.assert_index_equal(result, expected)