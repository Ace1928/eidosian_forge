import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interp_nonmono_raise(self):
    pytest.importorskip('scipy')
    s = Series([1, np.nan, 3], index=[0, 2, 1])
    msg = 'krogh interpolation requires that the index be monotonic'
    with pytest.raises(ValueError, match=msg):
        s.interpolate(method='krogh')