import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from .utils import (
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
@pytest.mark.parametrize('window', [5, 100])
@pytest.mark.parametrize('min_periods', [None, 5])
@pytest.mark.parametrize('axis', [lib.no_default, 1])
@pytest.mark.parametrize('method, kwargs', [('sum', {}), ('mean', {}), ('var', {'ddof': 0}), ('std', {'ddof': 0})])
def test_dataframe_window(data, window, min_periods, axis, method, kwargs):
    modin_df, pandas_df = create_test_dfs(data)
    if window > len(pandas_df):
        window = len(pandas_df)
    eval_general(modin_df, pandas_df, lambda df: getattr(df.rolling(window=window, min_periods=min_periods, win_type='triang', center=True, axis=axis), method)(**kwargs))