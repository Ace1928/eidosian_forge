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
def test_td_div_nan(self, nan):
    td = Timedelta(10, unit='d')
    result = td / nan
    assert result is NaT
    result = td // nan
    assert result is NaT