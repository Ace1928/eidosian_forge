from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('dropna, expected1, expected2', [(True, ['foo'], ['foo']), (False, ['foo'], [np.nan])])
def test_mode_mixeddtype(self, dropna, expected1, expected2):
    s = Series([1, 'foo', 'foo'])
    result = s.mode(dropna)
    expected = Series(expected1)
    tm.assert_series_equal(result, expected)
    s = Series([1, 'foo', 'foo', np.nan, np.nan, np.nan])
    result = s.mode(dropna)
    expected = Series(expected2, dtype=None if expected2 == ['foo'] else object)
    tm.assert_series_equal(result, expected)