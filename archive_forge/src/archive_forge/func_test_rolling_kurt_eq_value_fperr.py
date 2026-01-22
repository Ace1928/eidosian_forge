from functools import partial
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_rolling_kurt_eq_value_fperr(step):
    a = Series([1.1] * 15).rolling(window=10, step=step).kurt()
    assert (a[a.index >= 9] == -3).all()
    assert a[a.index < 9].isna().all()