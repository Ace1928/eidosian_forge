import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('signed_downcast', ['integer', 'signed'])
@pytest.mark.parametrize('data', [['1', 2, 3], [1, 2, 3], np.array(['1970-01-02', '1970-01-03', '1970-01-04'], dtype='datetime64[D]')])
def test_signed_downcast(data, signed_downcast):
    smallest_int_dtype = np.dtype(np.typecodes['Integer'][0])
    expected = np.array([1, 2, 3], dtype=smallest_int_dtype)
    res = to_numeric(data, downcast=signed_downcast)
    tm.assert_numpy_array_equal(res, expected)