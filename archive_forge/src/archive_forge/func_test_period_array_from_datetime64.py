import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import IncompatibleFrequency
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_period_array_from_datetime64():
    arr = np.array(['2020-01-01T00:00:00', '2020-02-02T00:00:00'], dtype='datetime64[ns]')
    result = PeriodArray._from_datetime64(arr, freq=MonthEnd(2))
    expected = period_array(['2020-01-01', '2020-02-01'], freq=MonthEnd(2))
    tm.assert_period_array_equal(result, expected)