import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.pandas.testing import assert_index_equal, assert_series_equal
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_insert_4407():
    data = {'col1': [1, 2, 3], 'col2': [2, 3, 4]}
    modin_df, pandas_df = create_test_dfs(data)

    def comparator(df1, df2):
        assert_series_equal(df1.dtypes, df2.dtypes, check_index=False)
        return df_equals(df1, df2)
    for idx, value in enumerate((pandas_df.to_numpy(), np.array([[1]] * 3), np.array([[1, 2, 3], [4, 5, 6]]))):
        expected_exception = None
        if idx == 0:
            expected_exception = ValueError('Expected a 1D array, got an array with shape (3, 2)')
        elif idx == 2:
            expected_exception = False
        eval_insert(modin_df, pandas_df, loc=0, col=f'test_col{idx}', value=value, comparator=lambda df1, df2: comparator(df1, df2), expected_exception=expected_exception)