import numpy as np
from pandas import (
import pandas._testing as tm
def test_asarray_homogeneous(self):
    df = DataFrame({'A': Categorical([1, 2]), 'B': Categorical([1, 2])})
    result = np.asarray(df)
    expected = np.array([[1, 1], [2, 2]], dtype='object')
    tm.assert_numpy_array_equal(result, expected)