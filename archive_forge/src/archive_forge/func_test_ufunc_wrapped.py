from __future__ import annotations
import warnings
import pytest
import numpy as np
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('ufunc', ['isreal', 'iscomplex', 'real', 'imag', 'angle', 'fix', 'i0', 'sinc', 'nan_to_num'])
def test_ufunc_wrapped(ufunc):
    """
    some np.ufuncs doesn't call __array_wrap__
    (or __array_ufunc__ starting from numpy v.1.13.0), it should work as below

    - da.ufunc(dd.Series) => da.Array
    - da.ufunc(pd.Series) => np.ndarray
    - np.ufunc(dd.Series) => np.ndarray
    - np.ufunc(pd.Series) => np.ndarray
    """
    from dask.array.utils import assert_eq as da_assert_eq
    if ufunc == 'fix':
        pytest.skip('fix calls floor in a way that we do not yet support')
    dafunc = getattr(da, ufunc)
    npfunc = getattr(np, ufunc)
    s = pd.Series(np.random.randint(1, 100, size=20), index=list('abcdefghijklmnopqrst'))
    ds = dd.from_pandas(s, 3)
    assert isinstance(dafunc(ds), da.Array)
    da_assert_eq(dafunc(ds), npfunc(s))
    assert isinstance(npfunc(ds), np.ndarray)
    np.testing.assert_equal(npfunc(ds), npfunc(s))
    assert isinstance(dafunc(s), np.ndarray)
    np.testing.assert_array_equal(dafunc(s), npfunc(s))
    df = pd.DataFrame({'A': np.random.randint(1, 100, size=20), 'B': np.random.randint(1, 100, size=20), 'C': np.abs(np.random.randn(20))}, index=list('abcdefghijklmnopqrst'))
    ddf = dd.from_pandas(df, 3)
    assert isinstance(dafunc(ddf), da.Array)
    da_assert_eq(dafunc(ddf), npfunc(df))
    assert isinstance(npfunc(ddf), np.ndarray)
    np.testing.assert_array_equal(npfunc(ddf), npfunc(df))
    assert isinstance(dafunc(df), np.ndarray)
    np.testing.assert_array_equal(dafunc(df), npfunc(df))