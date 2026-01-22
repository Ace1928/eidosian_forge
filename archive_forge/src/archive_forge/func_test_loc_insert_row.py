import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
@pytest.mark.parametrize('left, right', [(2, 1), (6, 1), (lambda df: 70, 1), (90, 70)])
def test_loc_insert_row(left, right):
    pandas_df = pandas.DataFrame([[1, 2, 3], [4, 5, 6]])
    modin_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])

    def _test_loc_rows(df):
        df.loc[left] = df.loc[right]
        return df
    expected_exception = None
    if right == 70:
        pytest.xfail(reason='https://github.com/modin-project/modin/issues/7024')
    eval_general(modin_df, pandas_df, _test_loc_rows, expected_exception=expected_exception)