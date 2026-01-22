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
@pytest.mark.parametrize('multi_col', ['col_multi_tree', 'col_multi_not_tree', 'col_index'])
@pytest.mark.parametrize('multi_idx', ['idx_multi_tree', 'idx_multi_not_tree', 'idx_index'])
def test_unstack_multiindex_types(multi_col, multi_idx):
    MAX_NROWS = MAX_NCOLS = 36
    pandas_df = pandas.DataFrame(test_data['int_data']).iloc[:MAX_NROWS, :MAX_NCOLS]
    modin_df = pd.DataFrame(test_data['int_data']).iloc[:MAX_NROWS, :MAX_NCOLS]

    def get_new_index(index, cond):
        if cond == 'col_multi_tree' or cond == 'idx_multi_tree':
            return generate_multiindex(len(index), nlevels=3, is_tree_like=True)
        elif cond == 'col_multi_not_tree' or cond == 'idx_multi_not_tree':
            return generate_multiindex(len(index), nlevels=3)
        else:
            return index
    pandas_df.columns = modin_df.columns = get_new_index(pandas_df.columns, multi_col)
    pandas_df.index = modin_df.index = get_new_index(pandas_df.index, multi_idx)
    df_equals(modin_df.unstack(), pandas_df.unstack())
    df_equals(modin_df.unstack(level=1), pandas_df.unstack(level=1))
    if multi_idx != 'idx_index':
        df_equals(modin_df.unstack(level=[0, 1]), pandas_df.unstack(level=[0, 1]))
        df_equals(modin_df.unstack(level=[0, 1, 2]), pandas_df.unstack(level=[0, 1, 2]))