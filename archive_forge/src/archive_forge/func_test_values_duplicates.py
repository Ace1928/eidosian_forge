import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_values_duplicates(self):
    df = DataFrame([[1, 2, 'a', 'b'], [1, 2, 'a', 'b']], columns=['one', 'one', 'two', 'two'])
    result = df.values
    expected = np.array([[1, 2, 'a', 'b'], [1, 2, 'a', 'b']], dtype=object)
    tm.assert_numpy_array_equal(result, expected)