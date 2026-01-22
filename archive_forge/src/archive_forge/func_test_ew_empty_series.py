import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['mean', 'std', 'var'])
def test_ew_empty_series(method):
    vals = Series([], dtype=np.float64)
    ewm = vals.ewm(3)
    result = getattr(ewm, method)()
    tm.assert_almost_equal(result, vals)