import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import IncompatibleFrequency
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_freq_deprecated():
    data = np.arange(5).astype(np.int64)
    msg = "The 'freq' keyword in the PeriodArray constructor is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = PeriodArray(data, freq='M')
    expected = PeriodArray(data, dtype='period[M]')
    tm.assert_equal(res, expected)