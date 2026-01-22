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
def test_pickle_dateoffset_odd_inputs(self):
    off = DateOffset(months=12)
    res = tm.round_trip_pickle(off)
    assert off == res
    base_dt = datetime(2020, 1, 1)
    assert base_dt + off == base_dt + res