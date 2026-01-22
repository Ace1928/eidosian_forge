import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
def test_multiindex_multi_key_not_supported(nogil, parallel, nopython):
    pytest.importorskip('numba')

    def numba_func(values, index):
        return 1
    df = DataFrame([{'A': 1, 'B': 2, 'C': 3}]).set_index(['A', 'B'])
    engine_kwargs = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
    with pytest.raises(NotImplementedError, match='more than 1 grouping labels'):
        df.groupby(['A', 'B']).transform(numba_func, engine='numba', engine_kwargs=engine_kwargs)