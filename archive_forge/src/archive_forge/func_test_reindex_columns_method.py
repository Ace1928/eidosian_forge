from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_columns_method(self):
    df = DataFrame(data=[[11, 12, 13], [21, 22, 23], [31, 32, 33]], index=[1, 2, 4], columns=[1, 2, 4], dtype=float)
    result = df.reindex(columns=range(6))
    expected = DataFrame(data=[[np.nan, 11, 12, np.nan, 13, np.nan], [np.nan, 21, 22, np.nan, 23, np.nan], [np.nan, 31, 32, np.nan, 33, np.nan]], index=[1, 2, 4], columns=range(6), dtype=float)
    tm.assert_frame_equal(result, expected)
    result = df.reindex(columns=range(6), method='ffill')
    expected = DataFrame(data=[[np.nan, 11, 12, 12, 13, 13], [np.nan, 21, 22, 22, 23, 23], [np.nan, 31, 32, 32, 33, 33]], index=[1, 2, 4], columns=range(6), dtype=float)
    tm.assert_frame_equal(result, expected)
    result = df.reindex(columns=range(6), method='bfill')
    expected = DataFrame(data=[[11, 11, 12, 13, 13, np.nan], [21, 21, 22, 23, 23, np.nan], [31, 31, 32, 33, 33, np.nan]], index=[1, 2, 4], columns=range(6), dtype=float)
    tm.assert_frame_equal(result, expected)