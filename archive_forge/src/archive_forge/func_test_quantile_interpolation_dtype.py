import numpy as np
import pytest
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import Timestamp
def test_quantile_interpolation_dtype(self):
    q = Series([1, 3, 4]).quantile(0.5, interpolation='lower')
    assert q == np.percentile(np.array([1, 3, 4]), 50)
    assert is_integer(q)
    q = Series([1, 3, 4]).quantile(0.5, interpolation='higher')
    assert q == np.percentile(np.array([1, 3, 4]), 50)
    assert is_integer(q)