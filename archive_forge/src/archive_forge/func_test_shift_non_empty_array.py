import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
@pytest.mark.parametrize('periods, indices', [[-4, [-1, -1]], [-1, [1, -1]], [0, [0, 1]], [1, [-1, 0]], [4, [-1, -1]]])
def test_shift_non_empty_array(self, data, periods, indices):
    subset = data[:2]
    result = subset.shift(periods)
    expected = subset.take(indices, allow_fill=True)
    tm.assert_extension_array_equal(result, expected)