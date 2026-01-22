from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_strftime_nat(self):
    arr = PeriodArray(PeriodIndex(['2019-01-01', NaT], dtype='period[D]'))
    result = arr.strftime('%Y-%m-%d')
    expected = np.array(['2019-01-01', np.nan], dtype=object)
    tm.assert_numpy_array_equal(result, expected)