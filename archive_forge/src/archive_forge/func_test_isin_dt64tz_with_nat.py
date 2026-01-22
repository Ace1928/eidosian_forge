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
def test_isin_dt64tz_with_nat(self):
    dti = date_range('2016-01-01', periods=3, tz='UTC')
    ser = Series(dti)
    ser[0] = NaT
    res = algos.isin(ser._values, [NaT])
    exp = np.array([True, False, False], dtype=bool)
    tm.assert_numpy_array_equal(res, exp)