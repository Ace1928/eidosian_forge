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
@pytest.mark.gpu
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_indexing_duplicate_axis(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_df.index = pandas_df.index = [i // 3 for i in range(len(modin_df))]
    assert any(modin_df.index.duplicated())
    assert any(pandas_df.index.duplicated())
    df_equals(modin_df.iloc[0], pandas_df.iloc[0])
    df_equals(modin_df.loc[0], pandas_df.loc[0])
    df_equals(modin_df.iloc[0, 0:4], pandas_df.iloc[0, 0:4])
    df_equals(modin_df.loc[0, modin_df.columns[0:4]], pandas_df.loc[0, pandas_df.columns[0:4]])