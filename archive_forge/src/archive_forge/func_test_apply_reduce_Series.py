from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_reduce_Series(float_frame):
    float_frame.iloc[::2, float_frame.columns.get_loc('A')] = np.nan
    expected = float_frame.mean(1)
    result = float_frame.apply(np.mean, axis=1)
    tm.assert_series_equal(result, expected)