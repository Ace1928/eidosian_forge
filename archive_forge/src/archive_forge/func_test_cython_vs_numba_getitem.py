import pytest
from pandas import (
import pandas._testing as tm
def test_cython_vs_numba_getitem(self, sort, nogil, parallel, nopython, numba_supported_reductions):
    func, kwargs = numba_supported_reductions
    df = DataFrame({'a': [3, 2, 3, 2], 'b': range(4), 'c': range(1, 5)})
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    gb = df.groupby('a', sort=sort)['c']
    result = getattr(gb, func)(engine='numba', engine_kwargs=engine_kwargs, **kwargs)
    expected = getattr(gb, func)(**kwargs)
    tm.assert_series_equal(result, expected)