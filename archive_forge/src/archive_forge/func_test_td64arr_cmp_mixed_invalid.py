from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_td64arr_cmp_mixed_invalid(self):
    rng = timedelta_range('1 days', periods=5)._data
    other = np.array([0, 1, 2, rng[3], Timestamp('2021-01-01')])
    result = rng == other
    expected = np.array([False, False, False, True, False])
    tm.assert_numpy_array_equal(result, expected)
    result = rng != other
    tm.assert_numpy_array_equal(result, ~expected)
    msg = 'Invalid comparison between|Cannot compare type|not supported between'
    with pytest.raises(TypeError, match=msg):
        rng < other
    with pytest.raises(TypeError, match=msg):
        rng > other
    with pytest.raises(TypeError, match=msg):
        rng <= other
    with pytest.raises(TypeError, match=msg):
        rng >= other