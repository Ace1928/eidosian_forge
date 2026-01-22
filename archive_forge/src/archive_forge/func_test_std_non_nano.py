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
def test_std_non_nano(self, unit):
    dti = pd.date_range('2016-01-01', periods=55, freq='D')
    arr = np.asarray(dti).astype(f'M8[{unit}]')
    dta = DatetimeArray._simple_new(arr, dtype=arr.dtype)
    res = dta.std()
    assert res._creso == dta._creso
    assert res == dti.std().floor(unit)