from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_floordiv_null_scalar(self):
    td = Timedelta(hours=3, minutes=4)
    assert td // np.nan is NaT
    assert np.isnan(td // NaT)
    assert np.isnan(td // np.timedelta64('NaT'))