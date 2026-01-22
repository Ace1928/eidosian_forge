import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
@pytest.mark.parametrize('a, b', [([1, None], [1, pd.NA]), ([None], [pd.NA]), ([None, np.nan], [pd.NA, pd.NA]), ([1, np.nan], [1, pd.NA]), ([np.nan], [pd.NA])])
def test_to_array_none_is_nan(a, b):
    result = pd.array(a, dtype='Float64')
    expected = pd.array(b, dtype='Float64')
    tm.assert_extension_array_equal(result, expected)