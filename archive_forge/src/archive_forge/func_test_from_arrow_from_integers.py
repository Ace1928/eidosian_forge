import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_from_arrow_from_integers():
    pa = pytest.importorskip('pyarrow')
    data = [0, 123456789, None, 2 ** 63 - 1, iNaT, -123456789]
    arr = pa.array(data)
    dtype = DatetimeTZDtype(unit='ns', tz='UTC')
    result = dtype.__from_arrow__(arr)
    expected = DatetimeArray._from_sequence(np.array(data, dtype='datetime64[ns]'))
    expected = expected.tz_localize('UTC')
    tm.assert_extension_array_equal(result, expected)
    result = dtype.__from_arrow__(pa.chunked_array([arr]))
    tm.assert_extension_array_equal(result, expected)