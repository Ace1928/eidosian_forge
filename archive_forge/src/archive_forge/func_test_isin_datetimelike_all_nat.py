from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
@pytest.mark.parametrize('dtype', ['m8[ns]', 'M8[ns]', 'M8[ns, UTC]', 'period[D]'])
def test_isin_datetimelike_all_nat(self, dtype):
    dta = date_range('2013-01-01', periods=3)._values
    arr = Series(dta.view('i8')).array.view(dtype)
    arr[0] = NaT
    result = algos.isin(arr, [NaT])
    expected = np.array([True, False, False], dtype=bool)
    tm.assert_numpy_array_equal(result, expected)