from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('other', [np.arange(1, 11), Index(np.arange(1, 11), np.int64), Index(range(1, 11), np.uint64), Index(range(1, 11), np.float64), pd.RangeIndex(1, 11)], ids=lambda x: type(x).__name__)
def test_tdi_rmul_arraylike(self, other, box_with_array):
    box = box_with_array
    tdi = TimedeltaIndex(['1 Day'] * 10)
    expected = timedelta_range('1 days', '10 days')._with_freq(None)
    tdi = tm.box_expected(tdi, box)
    xbox = get_upcast_box(tdi, other)
    expected = tm.box_expected(expected, xbox)
    result = other * tdi
    tm.assert_equal(result, expected)
    commute = tdi * other
    tm.assert_equal(commute, expected)