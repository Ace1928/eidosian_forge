import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.skipif(StorageFormat.get() != 'Pandas', reason="doesn't make sense for non-partitioned executions")
def test_corr_nans_in_different_partitions(self):
    modin_df, pandas_df = create_test_dfs({'a': [np.nan, 2, 3, 4, 5, 6], 'b': [3, 4, 2, 0, 7, 8]})
    modin_df = pd.concat([modin_df.iloc[:2], modin_df.iloc[2:4], modin_df.iloc[4:]])
    assert modin_df._query_compiler._modin_frame._partitions.shape == (3, 1)
    eval_general(modin_df, pandas_df, lambda df: df.corr())
    modin_df, pandas_df = create_test_dfs({'a': [1, 2, 3, 4, 5, np.nan], 'b': [3, 4, 2, 0, 7, 8]})
    modin_df = pd.concat([modin_df.iloc[:2], modin_df.iloc[2:4], modin_df.iloc[4:]])
    assert modin_df._query_compiler._modin_frame._partitions.shape == (3, 1)
    eval_general(modin_df, pandas_df, lambda df: df.corr())
    modin_df, pandas_df = create_test_dfs({'a': [np.nan, 2, 3, 4, 5, 6], 'b': [3, 4, 2, 0, 7, np.nan]})
    modin_df = pd.concat([modin_df.iloc[:2], modin_df.iloc[2:4], modin_df.iloc[4:]])
    assert modin_df._query_compiler._modin_frame._partitions.shape == (3, 1)
    eval_general(modin_df, pandas_df, lambda df: df.corr())
    modin_df, pandas_df = create_test_dfs({'a': [np.nan, 2, 3, np.nan, 5, 6], 'b': [3, 4, 2, 0, 7, np.nan]})
    modin_df = pd.concat([modin_df.iloc[:2], modin_df.iloc[2:4], modin_df.iloc[4:]])
    assert modin_df._query_compiler._modin_frame._partitions.shape == (3, 1)
    eval_general(modin_df, pandas_df, lambda df: df.corr())