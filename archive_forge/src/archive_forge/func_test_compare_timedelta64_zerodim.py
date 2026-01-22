from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_compare_timedelta64_zerodim(self, box_with_array):
    box = box_with_array
    xbox = box_with_array if box_with_array not in [Index, pd.array] else np.ndarray
    tdi = timedelta_range('2h', periods=4)
    other = np.array(tdi.to_numpy()[0])
    tdi = tm.box_expected(tdi, box)
    res = tdi <= other
    expected = np.array([True, False, False, False])
    expected = tm.box_expected(expected, xbox)
    tm.assert_equal(res, expected)