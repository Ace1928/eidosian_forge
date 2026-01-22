from datetime import (
import operator
import numpy as np
import pytest
from pandas import Timestamp
import pandas._testing as tm
def test_compare_zerodim_array(self, fixed_now_ts):
    ts = fixed_now_ts
    dt64 = np.datetime64('2016-01-01', 'ns')
    arr = np.array(dt64)
    assert arr.ndim == 0
    result = arr < ts
    assert result is np.bool_(True)
    result = arr > ts
    assert result is np.bool_(False)