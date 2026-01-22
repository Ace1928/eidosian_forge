import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_tz_dtype_mismatch_raises(self):
    arr = DatetimeArray._from_sequence(['2000'], dtype=DatetimeTZDtype(tz='US/Central'))
    with pytest.raises(TypeError, match='data is already tz-aware'):
        DatetimeArray._from_sequence(arr, dtype=DatetimeTZDtype(tz='UTC'))