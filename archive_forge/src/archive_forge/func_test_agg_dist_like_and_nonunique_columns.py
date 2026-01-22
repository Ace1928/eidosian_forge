from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_agg_dist_like_and_nonunique_columns():
    df = DataFrame({'A': [None, 2, 3], 'B': [1.0, np.nan, 3.0], 'C': ['foo', None, 'bar']})
    df.columns = ['A', 'A', 'C']
    result = df.agg({'A': 'count'})
    expected = df['A'].count()
    tm.assert_series_equal(result, expected)