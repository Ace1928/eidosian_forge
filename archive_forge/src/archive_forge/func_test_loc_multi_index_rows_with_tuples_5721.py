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
def test_loc_multi_index_rows_with_tuples_5721():
    arrays = [['bar', 'bar', 'baz', 'baz'], ['one', 'two', 'one', 'two']]
    ncols = 5
    index = pd.MultiIndex.from_tuples(zip(*arrays), names=['a', 'b'])
    data = np.arange(0, ncols * len(index)).reshape(len(index), ncols)
    modin_df, pandas_df = create_test_dfs(data, index=index)
    eval_general(modin_df, pandas_df, lambda df: df.loc['bar',])
    eval_general(modin_df, pandas_df, lambda df: df.loc['bar', 'two'])