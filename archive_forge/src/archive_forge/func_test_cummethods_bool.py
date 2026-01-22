import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('arg', [[False, False, False, True, True, False, False], [False, False, False, False, False, False, False]])
@pytest.mark.parametrize('func', [lambda x: x, lambda x: ~x], ids=['identity', 'inverse'])
@pytest.mark.parametrize('method', methods.keys())
def test_cummethods_bool(self, arg, func, method):
    ser = func(pd.Series(arg))
    ufunc = methods[method]
    exp_vals = ufunc(ser.values)
    expected = pd.Series(exp_vals)
    result = getattr(ser, method)()
    tm.assert_series_equal(result, expected)