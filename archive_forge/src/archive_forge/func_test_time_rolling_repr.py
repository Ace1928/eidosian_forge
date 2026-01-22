from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='different in dask-expr')
def test_time_rolling_repr():
    res = repr(dts.rolling('4s'))
    assert res == 'Rolling [window=4s,center=False,win_type=freq]'