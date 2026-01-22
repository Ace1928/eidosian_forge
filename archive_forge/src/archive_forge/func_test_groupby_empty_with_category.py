from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_groupby_empty_with_category():
    df = DataFrame({'A': [None] * 3, 'B': Categorical(['train', 'train', 'test'])})
    result = df.groupby('A').first()['B']
    expected = Series(Categorical([], categories=['test', 'train']), index=Series([], dtype='object', name='A'), name='B')
    tm.assert_series_equal(result, expected)