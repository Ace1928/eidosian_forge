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
def test_astype_non_nano_tznaive(self):
    dti = pd.date_range('2016-01-01', periods=3)
    res = dti.astype('M8[s]')
    assert res.dtype == 'M8[s]'
    dta = dti._data
    res = dta.astype('M8[s]')
    assert res.dtype == 'M8[s]'
    assert isinstance(res, pd.core.arrays.DatetimeArray)