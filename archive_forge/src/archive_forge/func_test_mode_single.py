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
@pytest.mark.parametrize('dt', np.typecodes['AllInteger'] + np.typecodes['Float'])
def test_mode_single(self, dt):
    exp_single = [1]
    data_single = [1]
    exp_multi = [1]
    data_multi = [1, 1]
    ser = Series(data_single, dtype=dt)
    exp = Series(exp_single, dtype=dt)
    tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
    tm.assert_series_equal(ser.mode(), exp)
    ser = Series(data_multi, dtype=dt)
    exp = Series(exp_multi, dtype=dt)
    tm.assert_numpy_array_equal(algos.mode(ser.values), exp.values)
    tm.assert_series_equal(ser.mode(), exp)