import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interp_all_good(self):
    pytest.importorskip('scipy')
    s = Series([1, 2, 3])
    result = s.interpolate(method='polynomial', order=1)
    tm.assert_series_equal(result, s)
    result = s.interpolate()
    tm.assert_series_equal(result, s)