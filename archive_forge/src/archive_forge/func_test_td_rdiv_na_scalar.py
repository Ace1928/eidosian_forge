from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_rdiv_na_scalar(self):
    td = Timedelta(10, unit='d')
    result = NaT / td
    assert np.isnan(result)
    result = None / td
    assert np.isnan(result)
    result = np.timedelta64('NaT') / td
    assert np.isnan(result)
    msg = "unsupported operand type\\(s\\) for /: 'numpy.datetime64' and 'Timedelta'"
    with pytest.raises(TypeError, match=msg):
        np.datetime64('NaT') / td
    msg = "unsupported operand type\\(s\\) for /: 'float' and 'Timedelta'"
    with pytest.raises(TypeError, match=msg):
        np.nan / td