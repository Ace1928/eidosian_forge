from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_dup_names_multi_agg():
    df = DataFrame([[0, 1], [2, 3]], columns=['a', 'a'])
    expected = DataFrame([[0, 1]], columns=['a', 'a'], index=['min'])
    result = df.agg(['min'])
    tm.assert_frame_equal(result, expected)