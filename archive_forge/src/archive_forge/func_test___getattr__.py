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
def test___getattr__(request, data):
    modin_df = pd.DataFrame(data)
    if 'empty_data' not in request.node.name:
        key = modin_df.columns[0]
        modin_df.__getattr__(key)
        col = modin_df.__getattr__('col1')
        assert isinstance(col, pd.Series)
        col = getattr(modin_df, 'col1')
        assert isinstance(col, pd.Series)
        df2 = modin_df.rename(index=str, columns={key: 'columns'})
        assert isinstance(df2.columns, pandas.Index)