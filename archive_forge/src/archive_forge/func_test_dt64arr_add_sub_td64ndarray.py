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
def test_dt64arr_add_sub_td64ndarray(self, tz_naive_fixture, box_with_array):
    tz = tz_naive_fixture
    dti = date_range('2016-01-01', periods=3, tz=tz)
    tdi = TimedeltaIndex(['-1 Day', '-1 Day', '-1 Day'])
    tdarr = tdi.values
    expected = date_range('2015-12-31', '2016-01-02', periods=3, tz=tz)
    dtarr = tm.box_expected(dti, box_with_array)
    expected = tm.box_expected(expected, box_with_array)
    result = dtarr + tdarr
    tm.assert_equal(result, expected)
    result = tdarr + dtarr
    tm.assert_equal(result, expected)
    expected = date_range('2016-01-02', '2016-01-04', periods=3, tz=tz)
    expected = tm.box_expected(expected, box_with_array)
    result = dtarr - tdarr
    tm.assert_equal(result, expected)
    msg = 'cannot subtract|(bad|unsupported) operand type for unary'
    with pytest.raises(TypeError, match=msg):
        tdarr - dtarr