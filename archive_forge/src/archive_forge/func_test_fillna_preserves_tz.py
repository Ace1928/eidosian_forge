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
@pytest.mark.parametrize('method', ['pad', 'backfill'])
def test_fillna_preserves_tz(self, method):
    dti = pd.date_range('2000-01-01', periods=5, freq='D', tz='US/Central')
    arr = DatetimeArray._from_sequence(dti, copy=True)
    arr[2] = pd.NaT
    fill_val = dti[1] if method == 'pad' else dti[3]
    expected = DatetimeArray._from_sequence([dti[0], dti[1], fill_val, dti[3], dti[4]], dtype=DatetimeTZDtype(tz='US/Central'))
    result = arr._pad_or_backfill(method=method)
    tm.assert_extension_array_equal(result, expected)
    assert arr[2] is pd.NaT
    assert dti[2] == pd.Timestamp('2000-01-03', tz='US/Central')