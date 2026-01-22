from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_rfloordiv_null_scalar(self):
    td = Timedelta(hours=3, minutes=3)
    assert np.isnan(td.__rfloordiv__(NaT))
    assert np.isnan(td.__rfloordiv__(np.timedelta64('NaT')))