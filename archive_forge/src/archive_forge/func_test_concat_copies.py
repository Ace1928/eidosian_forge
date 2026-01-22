import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ignore_index', [True, False])
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('axis', [0, 1])
def test_concat_copies(self, axis, order, ignore_index, using_copy_on_write):
    df = DataFrame(np.zeros((10, 5), dtype=np.float32, order=order))
    res = concat([df] * 5, axis=axis, ignore_index=ignore_index, copy=True)
    if not using_copy_on_write:
        for arr in res._iter_column_arrays():
            for arr2 in df._iter_column_arrays():
                assert not np.shares_memory(arr, arr2)