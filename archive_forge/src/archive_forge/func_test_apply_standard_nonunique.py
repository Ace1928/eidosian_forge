from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_standard_nonunique():
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['a', 'a', 'c'])
    result = df.apply(lambda s: s[0], axis=1)
    expected = Series([1, 4, 7], ['a', 'a', 'c'])
    tm.assert_series_equal(result, expected)
    result = df.T.apply(lambda s: s[0], axis=0)
    tm.assert_series_equal(result, expected)