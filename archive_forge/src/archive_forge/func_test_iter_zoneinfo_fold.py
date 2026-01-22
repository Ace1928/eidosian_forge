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
@pytest.mark.parametrize('tz', easts)
def test_iter_zoneinfo_fold(self, tz):
    utc_vals = np.array([1320552000, 1320555600, 1320559200, 1320562800], dtype=np.int64)
    utc_vals *= 1000000000
    dta = DatetimeArray._from_sequence(utc_vals).tz_localize('UTC').tz_convert(tz)
    left = dta[2]
    right = list(dta)[2]
    assert str(left) == str(right)
    assert left.utcoffset() == right.utcoffset()
    right2 = dta.astype(object)[2]
    assert str(left) == str(right2)
    assert left.utcoffset() == right2.utcoffset()