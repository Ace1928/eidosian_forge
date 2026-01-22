import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.arrays.boolean import coerce_to_array
def test_to_boolean_array_all_none():
    expected = BooleanArray(np.array([True, True, True]), np.array([True, True, True]))
    result = pd.array([None, None, None], dtype='boolean')
    tm.assert_extension_array_equal(result, expected)
    result = pd.array(np.array([None, None, None], dtype=object), dtype='boolean')
    tm.assert_extension_array_equal(result, expected)