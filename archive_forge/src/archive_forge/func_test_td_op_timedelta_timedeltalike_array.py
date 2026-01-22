from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('op', [operator.mul, ops.rmul, operator.truediv, ops.rdiv, ops.rsub])
@pytest.mark.parametrize('arr', [np.array([Timestamp('20130101 9:01'), Timestamp('20121230 9:02')]), np.array([Timestamp('2021-11-09 09:54:00'), Timedelta('1D')])])
def test_td_op_timedelta_timedeltalike_array(self, op, arr):
    msg = 'unsupported operand type|cannot use operands with types'
    with pytest.raises(TypeError, match=msg):
        op(arr, Timedelta('1D'))