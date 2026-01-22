import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
def test_value_counts_categorical():
    data = np.array(['a'] * 50000 + ['b'] * 10000 + ['c'] * 1000)
    random_state = np.random.RandomState(seed=42)
    random_state.shuffle(data)
    modin_df, pandas_df = create_test_dfs({'col1': data, 'col2': data}, dtype='category')
    if StorageFormat.get() == 'Hdk':

        def comparator(df1, df2):
            assert_dtypes_equal(df1, df2)
            assert_series_equal(df1._to_pandas(), df2, check_index=False, check_dtype=False)
    else:
        comparator = df_equals
    eval_general(modin_df, pandas_df, lambda df: df.value_counts(), comparator=comparator)