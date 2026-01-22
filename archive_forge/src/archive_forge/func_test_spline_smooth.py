import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_spline_smooth(self):
    pytest.importorskip('scipy')
    s = Series([1, 2, np.nan, 4, 5.1, np.nan, 7])
    assert s.interpolate(method='spline', order=3, s=0)[5] != s.interpolate(method='spline', order=3)[5]