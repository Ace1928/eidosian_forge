from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def test_rolling_mean_sum_floating_artifacts():
    sr = Series([1 / 3, 4, 0, 0, 0, 0, 0])
    r = sr.rolling(3)
    result = r.mean()
    assert (result[-3:] == 0).all()
    result = r.sum()
    assert (result[-3:] == 0).all()