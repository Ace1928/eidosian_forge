import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('minp', [1.0, 'foo', np.array([1, 2, 3])])
def test_invalid_minp(self, minp, regular):
    msg = "local variable 'minp' referenced before assignment|min_periods must be an integer"
    with pytest.raises(ValueError, match=msg):
        regular.rolling(window='1D', min_periods=minp)