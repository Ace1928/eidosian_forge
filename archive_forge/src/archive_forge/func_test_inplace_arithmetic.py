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
def test_inplace_arithmetic(self):
    data = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
    if self.array_cls is PeriodArray:
        arr = self.array_cls(data, dtype='period[D]')
    else:
        arr = self.index_cls(data, freq='D')._data
    expected = arr + pd.Timedelta(days=1)
    arr += pd.Timedelta(days=1)
    tm.assert_equal(arr, expected)
    expected = arr - pd.Timedelta(days=1)
    arr -= pd.Timedelta(days=1)
    tm.assert_equal(arr, expected)