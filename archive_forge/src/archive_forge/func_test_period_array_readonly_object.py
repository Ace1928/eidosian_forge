import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import IncompatibleFrequency
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_period_array_readonly_object():
    pa = period_array([pd.Period('2019-01-01')])
    arr = np.asarray(pa, dtype='object')
    arr.setflags(write=False)
    result = period_array(arr)
    tm.assert_period_array_equal(result, pa)
    result = pd.Series(arr)
    tm.assert_series_equal(result, pd.Series(pa))
    result = pd.DataFrame({'A': arr})
    tm.assert_frame_equal(result, pd.DataFrame({'A': pa}))