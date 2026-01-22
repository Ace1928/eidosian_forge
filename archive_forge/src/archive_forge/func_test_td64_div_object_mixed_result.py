from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64_div_object_mixed_result(self, box_with_array):
    orig = timedelta_range('1 Day', periods=3).insert(1, NaT)
    tdi = tm.box_expected(orig, box_with_array, transpose=False)
    other = np.array([orig[0], 1.5, 2.0, orig[2]], dtype=object)
    other = tm.box_expected(other, box_with_array, transpose=False)
    res = tdi / other
    expected = Index([1.0, np.timedelta64('NaT', 'ns'), orig[0], 1.5], dtype=object)
    expected = tm.box_expected(expected, box_with_array, transpose=False)
    if isinstance(expected, NumpyExtensionArray):
        expected = expected.to_numpy()
    tm.assert_equal(res, expected)
    if box_with_array is DataFrame:
        assert isinstance(res.iloc[1, 0], np.timedelta64)
    res = tdi // other
    expected = Index([1, np.timedelta64('NaT', 'ns'), orig[0], 1], dtype=object)
    expected = tm.box_expected(expected, box_with_array, transpose=False)
    if isinstance(expected, NumpyExtensionArray):
        expected = expected.to_numpy()
    tm.assert_equal(res, expected)
    if box_with_array is DataFrame:
        assert isinstance(res.iloc[1, 0], np.timedelta64)