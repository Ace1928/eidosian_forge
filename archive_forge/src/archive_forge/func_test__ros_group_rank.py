from statsmodels.compat.pandas import assert_series_equal, assert_frame_equal
from io import StringIO
from textwrap import dedent
import numpy as np
import numpy.testing as npt
import numpy
from numpy.testing import assert_equal
import pandas
import pytest
from statsmodels.imputation import ros
def test__ros_group_rank():
    df = pandas.DataFrame({'dl_idx': [1] * 12, 'params': list('AABCCCDE') + list('DCBA'), 'values': list(range(12))})
    result = ros._ros_group_rank(df, 'dl_idx', 'params')
    expected = pandas.Series([1, 2, 1, 1, 2, 3, 1, 1, 2, 4, 2, 3], name='rank')
    assert_series_equal(result.astype(int), expected.astype(int))