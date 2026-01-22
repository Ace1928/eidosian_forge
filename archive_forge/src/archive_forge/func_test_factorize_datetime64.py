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
def test_factorize_datetime64(self):
    v1 = Timestamp('20130101 09:00:00.00004')
    v2 = Timestamp('20130101')
    x = Series([v1, v1, v1, v2, v2, v1])
    codes, uniques = algos.factorize(x)
    exp = np.array([0, 0, 0, 1, 1, 0], dtype=np.intp)
    tm.assert_numpy_array_equal(codes, exp)
    exp = DatetimeIndex([v1, v2])
    tm.assert_index_equal(uniques, exp)
    codes, uniques = algos.factorize(x, sort=True)
    exp = np.array([1, 1, 1, 0, 0, 1], dtype=np.intp)
    tm.assert_numpy_array_equal(codes, exp)
    exp = DatetimeIndex([v2, v1])
    tm.assert_index_equal(uniques, exp)