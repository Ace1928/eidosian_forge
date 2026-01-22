from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('method,args,check_less_precise', rolling_method_args_check_less_precise)
@pytest.mark.parametrize('window', [1, 2, 4, 5])
@pytest.mark.parametrize('center', [True, False])
def test_rolling_methods(method, args, window, center, check_less_precise):
    if check_less_precise:
        check_less_precise = {'atol': 0.001, 'rtol': 0.001}
    else:
        check_less_precise = {}
    if method == 'count':
        min_periods = 0
    else:
        min_periods = None
    prolling = df.rolling(window, center=center, min_periods=min_periods)
    drolling = ddf.rolling(window, center=center, min_periods=min_periods)
    if method == 'apply':
        kwargs = {'raw': False}
    else:
        kwargs = {}
    assert_eq(getattr(prolling, method)(*args, **kwargs), getattr(drolling, method)(*args, **kwargs), **check_less_precise)
    prolling = df.a.rolling(window, center=center, min_periods=min_periods)
    drolling = ddf.a.rolling(window, center=center, min_periods=min_periods)
    assert_eq(getattr(prolling, method)(*args, **kwargs), getattr(drolling, method)(*args, **kwargs), **check_less_precise)