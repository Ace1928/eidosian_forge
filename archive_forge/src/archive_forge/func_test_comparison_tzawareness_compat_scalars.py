from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
def test_comparison_tzawareness_compat_scalars(self, comparison_op, box_with_array):
    op = comparison_op
    dr = date_range('2016-01-01', periods=6)
    dz = dr.tz_localize('US/Pacific')
    dr = tm.box_expected(dr, box_with_array)
    dz = tm.box_expected(dz, box_with_array)
    ts = Timestamp('2000-03-14 01:59')
    ts_tz = Timestamp('2000-03-14 01:59', tz='Europe/Amsterdam')
    assert np.all(dr > ts)
    msg = 'Invalid comparison between dtype=datetime64\\[ns.*\\] and Timestamp'
    if op not in [operator.eq, operator.ne]:
        with pytest.raises(TypeError, match=msg):
            op(dr, ts_tz)
    assert np.all(dz > ts_tz)
    if op not in [operator.eq, operator.ne]:
        with pytest.raises(TypeError, match=msg):
            op(dz, ts)
    if op not in [operator.eq, operator.ne]:
        with pytest.raises(TypeError, match=msg):
            op(ts, dz)