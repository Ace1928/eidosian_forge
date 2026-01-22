from datetime import timedelta
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_timedelta64_nan(self):
    td = Series([timedelta(days=i) for i in range(10)])
    td1 = td.copy()
    td1[0] = np.nan
    assert isna(td1[0])
    assert td1[0]._value == iNaT
    td1[0] = td[0]
    assert not isna(td1[0])
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        td1[1] = iNaT
    assert not isna(td1[1])
    assert td1.dtype == np.object_
    assert td1[1] == iNaT
    td1[1] = td[1]
    assert not isna(td1[1])
    td1[2] = NaT
    assert isna(td1[2])
    assert td1[2]._value == iNaT
    td1[2] = td[2]
    assert not isna(td1[2])
    td3 = np.timedelta64(timedelta(days=3))
    td7 = np.timedelta64(timedelta(days=7))
    td[(td > td3) & (td < td7)] = np.nan
    assert isna(td).sum() == 3