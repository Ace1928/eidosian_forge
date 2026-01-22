from datetime import datetime
import numpy as np
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_timezone_series_with_empty_series(self):
    time_index = date_range(datetime(2021, 1, 1, 1), datetime(2021, 1, 1, 10), freq='h', tz='Europe/Rome')
    s1 = Series(range(10), index=time_index)
    s2 = Series(index=time_index)
    msg = 'The behavior of array concatenation with empty entries is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s1.combine_first(s2)
    tm.assert_series_equal(result, s1)