import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interp_limit_bad_direction(self):
    s = Series([1, 3, np.nan, np.nan, np.nan, 11])
    msg = "Invalid limit_direction: expecting one of \\['forward', 'backward', 'both'\\], got 'abc'"
    with pytest.raises(ValueError, match=msg):
        s.interpolate(method='linear', limit=2, limit_direction='abc')
    with pytest.raises(ValueError, match=msg):
        s.interpolate(method='linear', limit_direction='abc')