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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='legacy, no longer supported in dask-expr')
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('redfunc', ['sum', 'prod', 'product', 'min', 'max', 'mean', 'var', 'std', 'all', 'any'])
def test_reductions_out(axis, redfunc):
    frame = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=[0, 1, 3])
    dsk_in = dd.from_pandas(frame, 3)
    out = dd.from_pandas(pd.Series([], dtype='float64'), 3)
    np_redfunc = getattr(np, redfunc)
    pd_redfunc = getattr(frame.__class__, redfunc)
    dsk_redfunc = getattr(dsk_in.__class__, redfunc)
    ctx = pytest.warns(FutureWarning, match="the 'out' keyword is deprecated")
    if redfunc in ['var', 'std']:
        with ctx:
            np_redfunc(dsk_in, axis=axis, ddof=1, out=out)
    elif NUMPY_GE_125 and redfunc == 'product' and (out is None):
        with pytest.warns(DeprecationWarning, match='`product` is deprecated'):
            np_redfunc(dsk_in, axis=axis, out=out)
    else:
        with ctx:
            np_redfunc(dsk_in, axis=axis, out=out)
    assert_eq(out, pd_redfunc(frame, axis=axis))
    with ctx:
        dsk_redfunc(dsk_in, axis=axis, split_every=False, out=out)
    assert_eq(out, pd_redfunc(frame, axis=axis))
    with pytest.warns(FutureWarning, match="the 'out' keyword is deprecated"):
        dsk_redfunc(dsk_in, axis=axis, split_every=2, out=out)
    assert_eq(out, pd_redfunc(frame, axis=axis))