import numpy as np
import pytest
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('dtype', [int, np.int32, np.int64, 'uint32', 'uint64'])
def test_astype_int(dtype):
    arr = period_array(['2000', '2001', None], freq='D')
    if np.dtype(dtype) != np.int64:
        with pytest.raises(TypeError, match="Do obj.astype\\('int64'\\)"):
            arr.astype(dtype)
        return
    result = arr.astype(dtype)
    expected = arr._ndarray.view('i8')
    tm.assert_numpy_array_equal(result, expected)