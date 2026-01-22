import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore')
def test_multiindex_one_key(nogil, parallel, nopython):
    pytest.importorskip('numba')

    def numba_func(values, index):
        return 1
    df = DataFrame([{'A': 1, 'B': 2, 'C': 3}]).set_index(['A', 'B'])
    engine_kwargs = {'nopython': nopython, 'nogil': nogil, 'parallel': parallel}
    result = df.groupby('A').transform(numba_func, engine='numba', engine_kwargs=engine_kwargs)
    expected = DataFrame([{'A': 1, 'B': 2, 'C': 1.0}]).set_index(['A', 'B'])
    tm.assert_frame_equal(result, expected)