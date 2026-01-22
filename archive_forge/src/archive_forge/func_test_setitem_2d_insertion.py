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
def test_setitem_2d_insertion():

    def build_value_picker(modin_value, pandas_value):
        """Build a function that returns either Modin or pandas DataFrame depending on the passed frame."""
        return lambda source_df, *args, **kwargs: modin_value if isinstance(source_df, (pd.DataFrame, pd.Series)) else pandas_value
    modin_df, pandas_df = create_test_dfs(test_data['int_data'])
    modin_value, pandas_value = create_test_dfs({'new_value1': np.arange(len(modin_df)), 'new_value2': np.arange(len(modin_df))})
    eval_setitem(modin_df, pandas_df, build_value_picker(modin_value, pandas_value), col=['new_value1', 'new_value2'])
    new_columns = ['new_value3', 'new_value4']
    modin_value.columns, pandas_value.columns = (new_columns, new_columns)
    eval_setitem(modin_df, pandas_df, build_value_picker(modin_value, pandas_value), col=['new_value4', 'new_value3'])
    new_columns = ['new_value5', 'new_value6']
    modin_value.columns, pandas_value.columns = (new_columns, new_columns)
    eval_setitem(modin_df, pandas_df, build_value_picker(modin_value, pandas_value), col=['__new_value5', '__new_value6'])
    eval_setitem(modin_df, pandas_df, build_value_picker(modin_value.iloc[:, [0]], pandas_value.iloc[:, [0]]), col=['new_value7', 'new_value8'], expected_exception=ValueError('Columns must be same length as key'))