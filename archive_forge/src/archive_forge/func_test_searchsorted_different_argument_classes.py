import numpy as np
import pytest
from pandas._libs.tslibs import IncompatibleFrequency
from pandas import (
import pandas._testing as tm
def test_searchsorted_different_argument_classes(self, listlike_box):
    pidx = PeriodIndex(['2014-01-01', '2014-01-02', '2014-01-03', '2014-01-04', '2014-01-05'], freq='D')
    result = pidx.searchsorted(listlike_box(pidx))
    expected = np.arange(len(pidx), dtype=result.dtype)
    tm.assert_numpy_array_equal(result, expected)
    result = pidx._data.searchsorted(listlike_box(pidx))
    tm.assert_numpy_array_equal(result, expected)