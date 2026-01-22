import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
@pytest.mark.parametrize('dtype', [np.int64, np.uint64])
def test_quantile_empty(dtype):
    arr = NumpyExtensionArray(np.array([], dtype=dtype))
    idx = pd.Index([0.0, 0.5])
    result = arr._quantile(idx, interpolation='linear')
    expected = NumpyExtensionArray(np.array([np.nan, np.nan]))
    tm.assert_extension_array_equal(result, expected)