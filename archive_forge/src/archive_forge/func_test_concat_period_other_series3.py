import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_period_other_series3(self):
    x = Series(pd.PeriodIndex(['2015-11-01', '2015-12-01'], freq='D'))
    y = Series(['A', 'B'])
    expected = Series([x[0], x[1], y[0], y[1]], dtype='object')
    result = concat([x, y], ignore_index=True)
    tm.assert_series_equal(result, expected)
    assert result.dtype == 'object'