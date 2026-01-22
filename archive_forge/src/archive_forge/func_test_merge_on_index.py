import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
@pytest.mark.parametrize('has_index_cache', [True, False])
def test_merge_on_index(has_index_cache):
    modin_df1, pandas_df1 = create_test_dfs({'idx_key1': [1, 2, 3, 4], 'idx_key2': [2, 3, 4, 5], 'idx_key3': [3, 4, 5, 6], 'data_col1': [10, 2, 3, 4], 'col_key1': [3, 4, 5, 6], 'col_key2': [3, 4, 5, 6]})
    modin_df1 = modin_df1.set_index(['idx_key1', 'idx_key2'])
    pandas_df1 = pandas_df1.set_index(['idx_key1', 'idx_key2'])
    modin_df2, pandas_df2 = create_test_dfs({'idx_key1': [4, 3, 2, 1], 'idx_key2': [5, 4, 3, 2], 'idx_key3': [6, 5, 4, 3], 'data_col2': [10, 2, 3, 4], 'col_key1': [6, 5, 4, 3], 'col_key2': [6, 5, 4, 3]})
    modin_df2 = modin_df2.set_index(['idx_key2', 'idx_key3'])
    pandas_df2 = pandas_df2.set_index(['idx_key2', 'idx_key3'])

    def setup_cache():
        if has_index_cache:
            modin_df1.index
            modin_df2.index
            assert modin_df1._query_compiler._modin_frame.has_index_cache
            assert modin_df2._query_compiler._modin_frame.has_index_cache
        else:
            modin_df1.index = modin_df1.index
            modin_df1._to_pandas()
            modin_df1._query_compiler._modin_frame.set_index_cache(None)
            modin_df2.index = modin_df2.index
            modin_df2._to_pandas()
            modin_df2._query_compiler._modin_frame.set_index_cache(None)
    for on in (['col_key1', 'idx_key1'], ['col_key1', 'idx_key2'], ['col_key1', 'idx_key3'], ['idx_key1'], ['idx_key2'], ['idx_key3']):
        setup_cache()
        eval_general((modin_df1, modin_df2), (pandas_df1, pandas_df2), lambda dfs: dfs[0].merge(dfs[1], on=on))
    for left_on, right_on in ((['idx_key1'], ['col_key1']), (['col_key1'], ['idx_key3']), (['idx_key1'], ['idx_key3']), (['idx_key2'], ['idx_key2']), (['col_key1', 'idx_key2'], ['col_key2', 'idx_key2'])):
        setup_cache()
        eval_general((modin_df1, modin_df2), (pandas_df1, pandas_df2), lambda dfs: dfs[0].merge(dfs[1], left_on=left_on, right_on=right_on))