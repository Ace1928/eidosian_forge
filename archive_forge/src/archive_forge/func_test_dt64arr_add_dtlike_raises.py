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
def test_dt64arr_add_dtlike_raises(self, tz_naive_fixture, box_with_array):
    tz = tz_naive_fixture
    dti = date_range('2016-01-01', periods=3, tz=tz)
    if tz is None:
        dti2 = dti.tz_localize('US/Eastern')
    else:
        dti2 = dti.tz_localize(None)
    dtarr = tm.box_expected(dti, box_with_array)
    assert_cannot_add(dtarr, dti.values)
    assert_cannot_add(dtarr, dti)
    assert_cannot_add(dtarr, dtarr)
    assert_cannot_add(dtarr, dti[0])
    assert_cannot_add(dtarr, dti[0].to_pydatetime())
    assert_cannot_add(dtarr, dti[0].to_datetime64())
    assert_cannot_add(dtarr, dti2[0])
    assert_cannot_add(dtarr, dti2[0].to_pydatetime())
    assert_cannot_add(dtarr, np.datetime64('2011-01-01', 'D'))