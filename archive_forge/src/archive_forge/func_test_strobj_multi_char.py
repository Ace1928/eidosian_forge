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
@pytest.mark.parametrize('dt', [str, object])
def test_strobj_multi_char(self, dt):
    exp = ['bar']
    data = ['foo'] * 2 + ['bar'] * 3
    ser = Series(data, dtype=dt)
    exp = Series(exp, dtype=dt)
    tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
    tm.assert_series_equal(ser.mode(), exp)