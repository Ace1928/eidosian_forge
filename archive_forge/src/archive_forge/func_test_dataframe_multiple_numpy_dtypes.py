import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_dataframe_multiple_numpy_dtypes():
    df = DataFrame({'a': [1, 2, 3], 'b': 1.5})
    arr = np.asarray(df)
    assert not np.shares_memory(arr, get_array(df, 'a'))
    assert arr.flags.writeable is True