import numpy as np
import pytest
from pandas._libs.tslibs import IncompatibleFrequency
from pandas import (
import pandas._testing as tm
def test_asof_preserves_bool_dtype(self):
    dti = date_range('2017-01-01', freq='MS', periods=4)
    ser = Series([True, False, True], index=dti[:-1])
    ts = dti[-1]
    res = ser.asof([ts])
    expected = Series([True], index=[ts])
    tm.assert_series_equal(res, expected)