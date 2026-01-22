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
@pytest.mark.parametrize('value', [1, np.int32(1), 1.0, 'str val', pandas.Timestamp('1/4/2018'), np.datetime64(0, 'ms'), True])
def test_loc_boolean_assignment_scalar_dtypes(value):
    modin_df, pandas_df = create_test_dfs({'a': [1, 2, 3], 'b': [3.0, 5.0, 6.0], 'c': ['a', 'b', 'c'], 'd': [1.0, 'c', 2.0], 'e': pandas.to_datetime(['1/1/2018', '1/2/2018', '1/3/2018']), 'f': [True, False, True]})
    modin_idx, pandas_idx = (pd.Series([False, True, True]), pandas.Series([False, True, True]))
    modin_df.loc[modin_idx] = value
    pandas_df.loc[pandas_idx] = value
    df_equals(modin_df, pandas_df)