import numpy as np
import pytest
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('jit', [True, False])
def test_numba_vs_cython_apply(self, jit, nogil, parallel, nopython, center, step):

    def f(x, *args):
        arg_sum = 0
        for arg in args:
            arg_sum += arg
        return np.mean(x) + arg_sum
    if jit:
        import numba
        f = numba.jit(f)
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    args = (2,)
    s = Series(range(10))
    result = s.rolling(2, center=center, step=step).apply(f, args=args, engine='numba', engine_kwargs=engine_kwargs, raw=True)
    expected = s.rolling(2, center=center, step=step).apply(f, engine='cython', args=args, raw=True)
    tm.assert_series_equal(result, expected)