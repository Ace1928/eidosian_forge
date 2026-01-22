import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
@pytest.mark.parametrize('copy', [False, None, True])
def test_frame_from_numpy_array(using_copy_on_write, copy, using_array_manager):
    arr = np.array([[1, 2], [3, 4]])
    df = DataFrame(arr, copy=copy)
    if using_copy_on_write and copy is not False or copy is True or (using_array_manager and copy is None):
        assert not np.shares_memory(get_array(df, 0), arr)
    else:
        assert np.shares_memory(get_array(df, 0), arr)