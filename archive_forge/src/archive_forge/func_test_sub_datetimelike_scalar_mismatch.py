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
def test_sub_datetimelike_scalar_mismatch(self):
    dti = pd.date_range('2016-01-01', periods=3)
    dta = dti._data.as_unit('us')
    ts = dta[0].as_unit('s')
    result = dta - ts
    expected = (dti - dti[0])._data.as_unit('us')
    assert result.dtype == 'm8[us]'
    tm.assert_extension_array_equal(result, expected)