import pytest
from pandas import (
import pandas._testing as tm
def test_axis_1_unsupported(self, numba_supported_reductions):
    func, kwargs = numba_supported_reductions
    df = DataFrame({'a': [3, 2, 3, 2], 'b': range(4), 'c': range(1, 5)})
    gb = df.groupby('a', axis=1)
    with pytest.raises(NotImplementedError, match='axis=1'):
        getattr(gb, func)(engine='numba', **kwargs)