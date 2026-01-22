import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('bad_raw', [None, 1, 0])
def test_rolling_apply_invalid_raw(bad_raw):
    with pytest.raises(ValueError, match='raw parameter must be `True` or `False`'):
        Series(range(3)).rolling(1).apply(len, raw=bad_raw)