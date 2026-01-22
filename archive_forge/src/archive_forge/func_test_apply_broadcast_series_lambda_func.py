from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_broadcast_series_lambda_func(int_frame_const_col):
    df = int_frame_const_col
    result = df.apply(lambda x: Series([1, 2, 3], index=list('abc')), axis=1, result_type='broadcast')
    expected = df.copy()
    tm.assert_frame_equal(result, expected)