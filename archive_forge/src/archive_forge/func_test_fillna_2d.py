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
def test_fillna_2d(self):
    dti = pd.date_range('2016-01-01', periods=6, tz='US/Pacific')
    dta = dti._data.reshape(3, 2).copy()
    dta[0, 1] = pd.NaT
    dta[1, 0] = pd.NaT
    res1 = dta._pad_or_backfill(method='pad')
    expected1 = dta.copy()
    expected1[1, 0] = dta[0, 0]
    tm.assert_extension_array_equal(res1, expected1)
    res2 = dta._pad_or_backfill(method='backfill')
    expected2 = dta.copy()
    expected2 = dta.copy()
    expected2[1, 0] = dta[2, 0]
    expected2[0, 1] = dta[1, 1]
    tm.assert_extension_array_equal(res2, expected2)
    dta2 = dta._from_backing_data(dta._ndarray.copy(order='F'))
    assert dta2._ndarray.flags['F_CONTIGUOUS']
    assert not dta2._ndarray.flags['C_CONTIGUOUS']
    tm.assert_extension_array_equal(dta, dta2)
    res3 = dta2._pad_or_backfill(method='pad')
    tm.assert_extension_array_equal(res3, expected1)
    res4 = dta2._pad_or_backfill(method='backfill')
    tm.assert_extension_array_equal(res4, expected2)
    df = pd.DataFrame(dta)
    res = df.ffill()
    expected = pd.DataFrame(expected1)
    tm.assert_frame_equal(res, expected)
    res = df.bfill()
    expected = pd.DataFrame(expected2)
    tm.assert_frame_equal(res, expected)