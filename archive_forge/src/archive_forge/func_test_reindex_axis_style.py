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
def test_reindex_axis_style(self):
    df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    expected = DataFrame({'A': [1, 2, np.nan], 'B': [4, 5, np.nan]}, index=[0, 1, 3])
    result = df.reindex([0, 1, 3])
    tm.assert_frame_equal(result, expected)
    result = df.reindex([0, 1, 3], axis=0)
    tm.assert_frame_equal(result, expected)
    result = df.reindex([0, 1, 3], axis='index')
    tm.assert_frame_equal(result, expected)