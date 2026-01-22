import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_dataframe_sub_numexpr_path(self):
    df = DataFrame({'A': np.random.default_rng(2).standard_normal(25000)})
    df.iloc[0:5] = np.nan
    expected = 1 - np.isnan(df.iloc[0:25])
    result = (1 - np.isnan(df)).iloc[0:25]
    tm.assert_frame_equal(result, expected)