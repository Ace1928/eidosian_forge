from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_idxmin_dt64index(self, unit):
    dti = DatetimeIndex(['NaT', '2015-02-08', 'NaT']).as_unit(unit)
    ser = Series([1.0, 2.0, np.nan], index=dti)
    msg = 'The behavior of Series.idxmin with all-NA values'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = ser.idxmin(skipna=False)
    assert res is NaT
    msg = 'The behavior of Series.idxmax with all-NA values'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = ser.idxmax(skipna=False)
    assert res is NaT
    df = ser.to_frame()
    msg = 'The behavior of DataFrame.idxmin with all-NA values'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = df.idxmin(skipna=False)
    assert res.dtype == f'M8[{unit}]'
    assert res.isna().all()
    msg = 'The behavior of DataFrame.idxmax with all-NA values'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = df.idxmax(skipna=False)
    assert res.dtype == f'M8[{unit}]'
    assert res.isna().all()