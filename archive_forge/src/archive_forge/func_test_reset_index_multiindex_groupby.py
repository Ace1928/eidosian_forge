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
@pytest.mark.parametrize('data', [test_data['int_data'], pytest.param(test_data['float_nan_data'], marks=pytest.mark.xfail(StorageFormat.get() == 'Hdk', reason='https://github.com/modin-project/modin/issues/2896'))])
def test_reset_index_multiindex_groupby(data):
    modin_df, pandas_df = create_test_dfs(data)
    modin_df.index = pd.MultiIndex.from_tuples([(i // 10, i // 5, i) for i in range(len(modin_df))])
    pandas_df.index = pandas.MultiIndex.from_tuples([(i // 10, i // 5, i) for i in range(len(pandas_df))])
    eval_general(modin_df, pandas_df, lambda df: df.reset_index().groupby(list(df.columns[:2])).count())