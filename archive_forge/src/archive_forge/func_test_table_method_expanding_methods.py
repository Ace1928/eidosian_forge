import numpy as np
import pytest
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_table_method_expanding_methods(self, axis, nogil, parallel, nopython, arithmetic_numba_supported_operators):
    method, kwargs = arithmetic_numba_supported_operators
    engine_kwargs = {'nogil': nogil, 'parallel': parallel, 'nopython': nopython}
    df = DataFrame(np.eye(3))
    expand_table = df.expanding(method='table', axis=axis)
    if method in ('var', 'std'):
        with pytest.raises(NotImplementedError, match=f'{method} not supported'):
            getattr(expand_table, method)(engine_kwargs=engine_kwargs, engine='numba', **kwargs)
    else:
        expand_single = df.expanding(method='single', axis=axis)
        result = getattr(expand_table, method)(engine_kwargs=engine_kwargs, engine='numba', **kwargs)
        expected = getattr(expand_single, method)(engine_kwargs=engine_kwargs, engine='numba', **kwargs)
        tm.assert_frame_equal(result, expected)