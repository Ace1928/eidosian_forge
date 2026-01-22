import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b', [(None, np.nan), (None, 0), (np.nan, 0)])
def test_assert_not_almost_equal_null(a, b):
    _assert_not_almost_equal(a, b)