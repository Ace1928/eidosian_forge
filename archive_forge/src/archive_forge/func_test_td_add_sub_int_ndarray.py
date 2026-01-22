from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_add_sub_int_ndarray(self):
    td = Timedelta('1 day')
    other = np.array([1])
    msg = "unsupported operand type\\(s\\) for \\+: 'Timedelta' and 'int'"
    with pytest.raises(TypeError, match=msg):
        td + np.array([1])
    msg = '|'.join(["unsupported operand type\\(s\\) for \\+: 'numpy.ndarray' and 'Timedelta'", 'Concatenation operation is not implemented for NumPy arrays'])
    with pytest.raises(TypeError, match=msg):
        other + td
    msg = "unsupported operand type\\(s\\) for -: 'Timedelta' and 'int'"
    with pytest.raises(TypeError, match=msg):
        td - other
    msg = "unsupported operand type\\(s\\) for -: 'numpy.ndarray' and 'Timedelta'"
    with pytest.raises(TypeError, match=msg):
        other - td