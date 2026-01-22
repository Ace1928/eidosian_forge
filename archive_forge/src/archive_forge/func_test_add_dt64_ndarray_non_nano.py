from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import (
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.errors import PerformanceWarning
from pandas import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import WeekDay
from pandas.tseries import offsets
from pandas.tseries.offsets import (
@pytest.mark.filterwarnings('ignore:Non-vectorized DateOffset being applied to Series or DatetimeIndex')
@pytest.mark.parametrize('unit', ['s', 'ms', 'us'])
def test_add_dt64_ndarray_non_nano(self, offset_types, unit):
    off = _create_offset(offset_types)
    dti = date_range('2016-01-01', periods=35, freq='D', unit=unit)
    result = (dti + off)._with_freq(None)
    exp_unit = unit
    if isinstance(off, Tick) and off._creso > dti._data._creso:
        exp_unit = Timedelta(off).unit
    expected = DatetimeIndex([x + off for x in dti]).as_unit(exp_unit)
    tm.assert_index_equal(result, expected)