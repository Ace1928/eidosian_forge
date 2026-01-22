import numpy as np
import pytest
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('grouper', ['None', 'groupby'])
@pytest.mark.parametrize('method', ['mean', 'sum'])
def test_cython_vs_numba(self, grouper, method, nogil, parallel, nopython, ignore_na, adjust):
    df = DataFrame({'B': range(4)})
    if grouper == 'None':
        grouper = lambda x: x
    else:
        df['A'] = ['a', 'b', 'a', 'b']
        grouper = lambda x: x.groupby('A')
    if method == 'sum':
        adjust = True
    ewm = grouper(df).ewm(com=1.0, adjust=adjust, ignore_na=ignore_na)
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    result = getattr(ewm, method)(engine='numba', engine_kwargs=engine_kwargs)
    expected = getattr(ewm, method)(engine='cython')
    tm.assert_frame_equal(result, expected)