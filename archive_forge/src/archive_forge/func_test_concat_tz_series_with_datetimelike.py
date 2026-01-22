import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_tz_series_with_datetimelike(self):
    x = [Timestamp('2011-01-01', tz='US/Eastern'), Timestamp('2011-02-01', tz='US/Eastern')]
    y = [pd.Timedelta('1 day'), pd.Timedelta('2 day')]
    result = concat([Series(x), Series(y)], ignore_index=True)
    tm.assert_series_equal(result, Series(x + y, dtype='object'))
    y = [pd.Period('2011-03', freq='M'), pd.Period('2011-04', freq='M')]
    result = concat([Series(x), Series(y)], ignore_index=True)
    tm.assert_series_equal(result, Series(x + y, dtype='object'))