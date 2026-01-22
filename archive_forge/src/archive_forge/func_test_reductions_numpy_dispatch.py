from __future__ import annotations
import contextlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_scalar
import dask.dataframe as dd
from dask.array.numpy_compat import NUMPY_GE_125
from dask.dataframe._compat import (
from dask.dataframe.utils import (
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('redfunc', ['sum', 'prod', 'product', 'min', 'max', 'mean', 'var', 'std', 'all', 'any'])
def test_reductions_numpy_dispatch(axis, redfunc):
    pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=[0, 1, 3])
    df = dd.from_pandas(pdf, 3)
    np_redfunc = getattr(np, redfunc)
    if redfunc in ('var', 'std'):
        expect = np_redfunc(pdf, axis=axis, ddof=1)
        actual = np_redfunc(df, axis=axis, ddof=1)
    elif NUMPY_GE_125 and redfunc == 'product':
        expect = np_redfunc(pdf, axis=axis)
        with pytest.warns(DeprecationWarning, match='`product` is deprecated'):
            actual = np_redfunc(df, axis=axis)
    else:
        expect = np_redfunc(pdf, axis=axis)
        actual = np_redfunc(df, axis=axis)
    assert_eq(expect, actual)