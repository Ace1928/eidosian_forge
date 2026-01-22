import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('ufunc', [np.abs, np.sign])
@pytest.mark.filterwarnings('ignore:invalid value encountered in sign:RuntimeWarning')
def test_ufuncs_single_int(ufunc):
    a = pd.array([1, 2, -3, np.nan])
    result = ufunc(a)
    expected = pd.array(ufunc(a.astype(float)), dtype='Int64')
    tm.assert_extension_array_equal(result, expected)
    s = pd.Series(a)
    result = ufunc(s)
    expected = pd.Series(pd.array(ufunc(a.astype(float)), dtype='Int64'))
    tm.assert_series_equal(result, expected)