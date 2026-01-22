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
@pytest.mark.parametrize('meth', ['time', 'timetz', 'date'])
def test_time_date(self, dta_dti, meth):
    dta, dti = dta_dti
    result = getattr(dta, meth)
    expected = getattr(dti, meth)
    tm.assert_numpy_array_equal(result, expected)