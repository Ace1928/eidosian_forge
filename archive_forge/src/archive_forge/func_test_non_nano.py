from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_non_nano(self, unit, dtype):
    arr = np.arange(5, dtype=np.int64).view(f'M8[{unit}]')
    dta = DatetimeArray._simple_new(arr, dtype=dtype)
    assert dta.dtype == dtype
    assert dta[0].unit == unit
    assert tz_compare(dta.tz, dta[0].tz)
    assert (dta[0] == dta[:1]).all()