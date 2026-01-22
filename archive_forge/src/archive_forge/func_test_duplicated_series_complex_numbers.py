from itertools import product
import numpy as np
import pytest
from pandas._libs import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_duplicated_series_complex_numbers(dtype):
    expected = Series([False, False, False, True, False, False, False, True, False, True], dtype=bool)
    result = Series([np.nan + np.nan * 1j, 0, 1j, 1j, 1, 1 + 1j, 1 + 2j, 1 + 1j, np.nan, np.nan + np.nan * 1j], dtype=dtype).duplicated()
    tm.assert_series_equal(result, expected)