import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('start, periods', [('2010-03-31', 1), ('2010-03-30', 2)])
def test_first_with_first_day_last_of_month(self, frame_or_series, start, periods):
    x = frame_or_series([1] * 100, index=bdate_range(start, periods=100))
    with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
        result = x.first('1ME')
    expected = frame_or_series([1] * periods, index=bdate_range(start, periods=periods))
    tm.assert_equal(result, expected)