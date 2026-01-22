import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
def test_rpow_one_to_na():
    arr = pd.array([np.nan, np.nan], dtype='Int64')
    result = np.array([1.0, 2.0]) ** arr
    expected = pd.array([1.0, np.nan], dtype='Float64')
    tm.assert_extension_array_equal(result, expected)