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
@pytest.mark.parametrize('other', [datetime(2016, 1, 1), Timestamp('2016-01-01'), np.datetime64('2016-01-01')])
@pytest.mark.filterwarnings('ignore:elementwise comp:DeprecationWarning')
def test_scalar_comparison_tzawareness(self, comparison_op, other, tz_aware_fixture, box_with_array):
    op = comparison_op
    tz = tz_aware_fixture
    dti = date_range('2016-01-01', periods=2, tz=tz)
    dtarr = tm.box_expected(dti, box_with_array)
    xbox = get_upcast_box(dtarr, other, True)
    if op in [operator.eq, operator.ne]:
        exbool = op is operator.ne
        expected = np.array([exbool, exbool], dtype=bool)
        expected = tm.box_expected(expected, xbox)
        result = op(dtarr, other)
        tm.assert_equal(result, expected)
        result = op(other, dtarr)
        tm.assert_equal(result, expected)
    else:
        msg = f'Invalid comparison between dtype=datetime64\\[ns, .*\\] and {type(other).__name__}'
        with pytest.raises(TypeError, match=msg):
            op(dtarr, other)
        with pytest.raises(TypeError, match=msg):
            op(other, dtarr)