from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('dropna, expected1, expected2', [(True, [2 ** 63], [1, 2 ** 63]), (False, [2 ** 63], [1, 2 ** 63])])
def test_mode_intoverflow(self, dropna, expected1, expected2):
    s = Series([1, 2 ** 63, 2 ** 63], dtype=np.uint64)
    result = s.mode(dropna)
    expected1 = Series(expected1, dtype=np.uint64)
    tm.assert_series_equal(result, expected1)
    s = Series([1, 2 ** 63], dtype=np.uint64)
    result = s.mode(dropna)
    expected2 = Series(expected2, dtype=np.uint64)
    tm.assert_series_equal(result, expected2)