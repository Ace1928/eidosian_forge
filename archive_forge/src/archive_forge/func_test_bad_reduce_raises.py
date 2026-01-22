import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
def test_bad_reduce_raises():
    arr = np.array([1, 2, 3], dtype='int64')
    arr = NumpyExtensionArray(arr)
    msg = 'cannot perform not_a_method with type int'
    with pytest.raises(TypeError, match=msg):
        arr._reduce(msg)