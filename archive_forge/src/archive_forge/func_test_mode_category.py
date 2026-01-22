from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('dropna, expected1, expected2, expected3', [(True, Categorical([1, 2], categories=[1, 2]), Categorical(['a'], categories=[1, 'a']), Categorical([3, 1], categories=[3, 2, 1], ordered=True)), (False, Categorical([np.nan], categories=[1, 2]), Categorical([np.nan, 'a'], categories=[1, 'a']), Categorical([np.nan, 3, 1], categories=[3, 2, 1], ordered=True))])
def test_mode_category(self, dropna, expected1, expected2, expected3):
    s = Series(Categorical([1, 2, np.nan, np.nan]))
    result = s.mode(dropna)
    expected1 = Series(expected1, dtype='category')
    tm.assert_series_equal(result, expected1)
    s = Series(Categorical([1, 'a', 'a', np.nan, np.nan]))
    result = s.mode(dropna)
    expected2 = Series(expected2, dtype='category')
    tm.assert_series_equal(result, expected2)
    s = Series(Categorical([1, 1, 2, 3, 3, np.nan, np.nan], categories=[3, 2, 1], ordered=True))
    result = s.mode(dropna)
    expected3 = Series(expected3, dtype='category')
    tm.assert_series_equal(result, expected3)