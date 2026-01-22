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
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
@pytest.mark.parametrize('sortorder', [0, 3, 5])
def test_multiindex_from_frame(data, sortorder):
    modin_df, pandas_df = create_test_dfs(data)

    def call_from_frame(df):
        if type(df).__module__.startswith('pandas'):
            return pandas.MultiIndex.from_frame(df, sortorder)
        else:
            return pd.MultiIndex.from_frame(df, sortorder)
    eval_general(modin_df, pandas_df, call_from_frame, comparator=assert_index_equal)