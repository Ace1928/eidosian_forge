import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('test_ddof', [None, 0, 1, 2, 3])
def test_cov_ddof(self, test_ddof):
    np_array1 = np.random.default_rng(2).random(10)
    np_array2 = np.random.default_rng(2).random(10)
    df = DataFrame({0: np_array1, 1: np_array2})
    result = df.cov(ddof=test_ddof)
    expected_np = np.cov(np_array1, np_array2, ddof=test_ddof)
    expected = DataFrame(expected_np)
    tm.assert_frame_equal(result, expected)