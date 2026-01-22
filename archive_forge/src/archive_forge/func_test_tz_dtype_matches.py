import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_tz_dtype_matches(self):
    dtype = DatetimeTZDtype(tz='US/Central')
    arr = DatetimeArray._from_sequence(['2000'], dtype=dtype)
    result = DatetimeArray._from_sequence(arr, dtype=dtype)
    tm.assert_equal(arr, result)