import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
@pytest.mark.parametrize('kwargs', [{'right_index': True, 'left_index': True}, {'left_on': 'left_time', 'right_index': True}, {'left_index': True, 'right_on': 'right'}])
def test_merge_asof_index_behavior(kwargs):
    index = Index([1, 5, 10], name='test')
    left = pd.DataFrame({'left': ['a', 'b', 'c'], 'left_time': [1, 4, 10]}, index=index)
    right = pd.DataFrame({'right': [1, 2, 3, 6, 7]}, index=[1, 2, 3, 6, 7])
    result = merge_asof(left, right, **kwargs)
    expected = pd.DataFrame({'left': ['a', 'b', 'c'], 'left_time': [1, 4, 10], 'right': [1, 3, 7]}, index=index)
    tm.assert_frame_equal(result, expected)