from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_mod_int(self, box_with_array):
    tdi = timedelta_range('1 ns', '10 ns', periods=10)
    tdarr = tm.box_expected(tdi, box_with_array)
    expected = TimedeltaIndex(['1 ns', '0 ns'] * 5)
    expected = tm.box_expected(expected, box_with_array)
    result = tdarr % 2
    tm.assert_equal(result, expected)
    msg = 'Cannot divide int by'
    with pytest.raises(TypeError, match=msg):
        2 % tdarr
    result = divmod(tdarr, 2)
    tm.assert_equal(result[1], expected)
    tm.assert_equal(result[0], tdarr // 2)