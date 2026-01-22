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
def test_array_tz(self, arr1d):
    arr = arr1d
    dti = self.index_cls(arr1d)
    expected = dti.asi8.view('M8[ns]')
    result = np.array(arr, dtype='M8[ns]')
    tm.assert_numpy_array_equal(result, expected)
    result = np.array(arr, dtype='datetime64[ns]')
    tm.assert_numpy_array_equal(result, expected)
    result = np.array(arr, dtype='M8[ns]', copy=False)
    assert result.base is expected.base
    assert result.base is not None
    result = np.array(arr, dtype='datetime64[ns]', copy=False)
    assert result.base is expected.base
    assert result.base is not None