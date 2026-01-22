from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.missing import NA
from pandas.core.dtypes.common import is_scalar
import pandas as pd
import pandas._testing as tm
def test_binary_input_not_dunder():
    a = np.array([1, 2, 3])
    expected = np.array([NA, NA, NA], dtype=object)
    result = np.logaddexp(a, NA)
    tm.assert_numpy_array_equal(result, expected)
    result = np.logaddexp(NA, a)
    tm.assert_numpy_array_equal(result, expected)
    assert np.logaddexp(NA, NA) is NA
    result = np.modf(NA, NA)
    assert len(result) == 2
    assert all((x is NA for x in result))