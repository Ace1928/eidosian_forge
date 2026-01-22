from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.xfail(DASK_EXPR_ENABLED, reason='this works in dask-expr')
def test_groupby_rolling_with_integer_window_raises():
    df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4, 5, 6], 'C': ['a', 'a', 'a', 'b', 'b', 'a', 'b']})
    ddf = dd.from_pandas(df, npartitions=2)
    with pytest.raises(ValueError, match='``window`` must be a ``freq``'):
        ddf.groupby('C').rolling(2).sum()