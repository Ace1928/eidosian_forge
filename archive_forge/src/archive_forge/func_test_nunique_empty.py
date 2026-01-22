from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_nunique_empty():
    df = DataFrame(columns=['a', 'b', 'c'])
    result = df.nunique()
    expected = Series(0, index=df.columns)
    tm.assert_series_equal(result, expected)
    result = df.T.nunique()
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)