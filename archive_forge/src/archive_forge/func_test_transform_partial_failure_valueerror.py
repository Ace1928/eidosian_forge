import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_transform_partial_failure_valueerror():

    def noop(x):
        return x

    def raising_op(_):
        raise ValueError
    ser = Series(3 * [object])
    msg = 'Transform function failed'
    with pytest.raises(ValueError, match=msg):
        ser.transform([noop, raising_op])
    with pytest.raises(ValueError, match=msg):
        ser.transform({'A': raising_op, 'B': noop})
    with pytest.raises(ValueError, match=msg):
        ser.transform({'A': [raising_op], 'B': [noop]})
    with pytest.raises(ValueError, match=msg):
        ser.transform({'A': [noop, raising_op], 'B': [noop]})