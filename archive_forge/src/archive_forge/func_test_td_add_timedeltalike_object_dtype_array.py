from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('op', [operator.add, ops.radd])
def test_td_add_timedeltalike_object_dtype_array(self, op):
    arr = np.array([Timestamp('20130101 9:01'), Timestamp('20121230 9:02')])
    exp = np.array([Timestamp('20130102 9:01'), Timestamp('20121231 9:02')])
    res = op(arr, Timedelta('1D'))
    tm.assert_numpy_array_equal(res, exp)