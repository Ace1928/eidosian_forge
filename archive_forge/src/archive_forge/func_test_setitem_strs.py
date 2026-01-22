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
def test_setitem_strs(self, arr1d):
    expected = arr1d.copy()
    expected[[0, 1]] = arr1d[-2:]
    result = arr1d.copy()
    result[:2] = [str(x) for x in arr1d[-2:]]
    tm.assert_equal(result, expected)
    expected = arr1d.copy()
    expected[0] = arr1d[-1]
    result = arr1d.copy()
    result[0] = str(arr1d[-1])
    tm.assert_equal(result, expected)