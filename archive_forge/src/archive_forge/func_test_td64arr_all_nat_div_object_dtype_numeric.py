from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_all_nat_div_object_dtype_numeric(self, box_with_array):
    tdi = TimedeltaIndex([NaT, NaT])
    left = tm.box_expected(tdi, box_with_array)
    right = np.array([2, 2.0], dtype=object)
    tdnat = np.timedelta64('NaT', 'ns')
    expected = Index([tdnat] * 2, dtype=object)
    if box_with_array is not Index:
        expected = tm.box_expected(expected, box_with_array).astype(object)
        if box_with_array in [Series, DataFrame]:
            msg = "The 'downcast' keyword in fillna is deprecated"
            with tm.assert_produces_warning(FutureWarning, match=msg):
                expected = expected.fillna(tdnat, downcast=False)
    result = left / right
    tm.assert_equal(result, expected)
    result = left // right
    tm.assert_equal(result, expected)