from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_empty_list_reduce():
    df = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], columns=['a', 'b'])
    result = df.apply(lambda x: [], result_type='reduce')
    expected = Series({'a': [], 'b': []}, dtype=object)
    tm.assert_series_equal(result, expected)