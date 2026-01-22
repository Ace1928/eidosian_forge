from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('func, expected_values', [(Series.nunique, [1, 1, 2]), (Series.count, [1, 2, 2])])
def test_groupby_agg_categorical_columns(func, expected_values):
    df = DataFrame({'id': [0, 1, 2, 3, 4], 'groups': [0, 1, 1, 2, 2], 'value': Categorical([0, 0, 0, 0, 1])}).set_index('id')
    result = df.groupby('groups').agg(func)
    expected = DataFrame({'value': expected_values}, index=Index([0, 1, 2], name='groups'))
    tm.assert_frame_equal(result, expected)