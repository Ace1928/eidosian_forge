from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('nan', [np.nan, np.float64('NaN'), float('nan')])
@pytest.mark.parametrize('op', [operator.mul, ops.rmul])
def test_td_mul_nan(self, op, nan):
    td = Timedelta(10, unit='d')
    result = op(td, nan)
    assert result is NaT