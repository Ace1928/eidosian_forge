import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(Index([1.0, 1.1]), Index([1.0, 1.100001])), (Series([1.0, 1.1]), Series([1.0, 1.100001])), (np.array([1.1, 2.000001]), np.array([1.1, 2.0])), (DataFrame({'a': [1.0, 1.1]}), DataFrame({'a': [1.0, 1.100001]}))])
def test_assert_almost_equal_pandas(a, b):
    _assert_almost_equal_both(a, b)