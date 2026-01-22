from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_add_sub_td64_array(self, box_with_array):
    box = box_with_array
    dti = pd.date_range('2016-01-01', periods=3)
    tdi = dti - dti.shift(1)
    tdarr = tdi.values
    expected = 2 * tdi
    tdi = tm.box_expected(tdi, box)
    expected = tm.box_expected(expected, box)
    result = tdi + tdarr
    tm.assert_equal(result, expected)
    result = tdarr + tdi
    tm.assert_equal(result, expected)
    expected_sub = 0 * tdi
    result = tdi - tdarr
    tm.assert_equal(result, expected_sub)
    result = tdarr - tdi
    tm.assert_equal(result, expected_sub)