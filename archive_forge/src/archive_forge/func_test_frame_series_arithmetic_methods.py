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
def test_frame_series_arithmetic_methods():
    pdf1 = pd.DataFrame({'A': np.arange(10), 'B': [np.nan, 1, 2, 3, 4] * 2, 'C': [np.nan] * 10, 'D': np.arange(10)}, index=list('abcdefghij'), columns=list('ABCD'))
    pdf2 = pd.DataFrame(np.random.randn(10, 4), index=list('abcdefghjk'), columns=list('ABCX'))
    ps1 = pdf1.A
    ps2 = pdf2.A
    ps3 = pd.Series(np.random.randn(10), index=list('ABCDXabcde'))
    ddf1 = dd.from_pandas(pdf1, 2)
    ddf2 = dd.from_pandas(pdf2, 2)
    ds1 = ddf1.A
    ds2 = ddf2.A
    if DASK_EXPR_ENABLED:
        s = 4
    else:
        s = dd.core.Scalar({('s', 0): 4}, 's', 'i8')
    for l, r, el, er in [(ddf1, ddf2, pdf1, pdf2), (ds1, ds2, ps1, ps2), (ddf1.repartition(['a', 'f', 'j']), ddf2, pdf1, pdf2), (ds1.repartition(['a', 'b', 'f', 'j']), ds2, ps1, ps2), (ddf1, ddf2.repartition(['a', 'k']), pdf1, pdf2), (ds1, ds2.repartition(['a', 'b', 'd', 'h', 'k']), ps1, ps2), (ddf1, 3, pdf1, 3), (ds1, 3, ps1, 3), (ddf1, s, pdf1, 4), (ds1, s, ps1, 4)]:
        assert_eq(l, el)
        assert_eq(r, er)
        assert_eq(l.add(r, fill_value=0), el.add(er, fill_value=0))
        assert_eq(l.sub(r, fill_value=0), el.sub(er, fill_value=0))
        assert_eq(l.mul(r, fill_value=0), el.mul(er, fill_value=0))
        assert_eq(l.div(r, fill_value=0), el.div(er, fill_value=0))
        assert_eq(l.divide(r, fill_value=0), el.divide(er, fill_value=0))
        assert_eq(l.truediv(r, fill_value=0), el.truediv(er, fill_value=0))
        assert_eq(l.floordiv(r, fill_value=1), el.floordiv(er, fill_value=1))
        assert_eq(l.pow(r, fill_value=0), el.pow(er, fill_value=0))
        assert_eq(l.mod(r, fill_value=0), el.mod(er, fill_value=0))
        assert_eq(l.radd(r, fill_value=0), el.radd(er, fill_value=0))
        assert_eq(l.rsub(r, fill_value=0), el.rsub(er, fill_value=0))
        assert_eq(l.rmul(r, fill_value=0), el.rmul(er, fill_value=0))
        assert_eq(l.rdiv(r, fill_value=0), el.rdiv(er, fill_value=0))
        assert_eq(l.rtruediv(r, fill_value=0), el.rtruediv(er, fill_value=0))
        assert_eq(l.rpow(r, fill_value=0), el.rpow(er, fill_value=0))
        assert_eq(l.rmod(r, fill_value=0), el.rmod(er, fill_value=0))
    for l, r, el, er in [(ddf1, ds2, pdf1, ps2), (ddf1, ddf2.X, pdf1, pdf2.X)]:
        assert_eq(l, el)
        assert_eq(r, er)
        assert_eq(l.add(r, axis=0), el.add(er, axis=0))
        assert_eq(l.sub(r, axis=0), el.sub(er, axis=0))
        assert_eq(l.mul(r, axis=0), el.mul(er, axis=0))
        assert_eq(l.div(r, axis=0), el.div(er, axis=0))
        assert_eq(l.divide(r, axis=0), el.divide(er, axis=0))
        assert_eq(l.truediv(r, axis=0), el.truediv(er, axis=0))
        assert_eq(l.floordiv(r, axis=0), el.floordiv(er, axis=0))
        assert_eq(l.mod(r, axis=0), el.mod(er, axis=0))
        assert_eq(l.pow(r, axis=0), el.pow(er, axis=0))
        assert_eq(l.radd(r, axis=0), el.radd(er, axis=0))
        assert_eq(l.rsub(r, axis=0), el.rsub(er, axis=0))
        assert_eq(l.rmul(r, axis=0), el.rmul(er, axis=0))
        assert_eq(l.rdiv(r, axis=0), el.rdiv(er, axis=0))
        assert_eq(l.rtruediv(r, axis=0), el.rtruediv(er, axis=0))
        assert_eq(l.rmod(r, axis=0), el.rmod(er, axis=0))
        assert_eq(l.rpow(r, axis=0), el.rpow(er, axis=0))
        pytest.raises(ValueError, lambda l=l, r=r: l.add(r, axis=1))
    for l, r, el, er in [(ddf1, pdf2, pdf1, pdf2), (ddf1, ps3, pdf1, ps3)]:
        assert_eq(l, el)
        assert_eq(r, er)
        for axis in [0, 1, 'index', 'columns']:
            assert_eq(l.add(r, axis=axis), el.add(er, axis=axis))
            assert_eq(l.sub(r, axis=axis), el.sub(er, axis=axis))
            assert_eq(l.mul(r, axis=axis), el.mul(er, axis=axis))
            assert_eq(l.div(r, axis=axis), el.div(er, axis=axis))
            assert_eq(l.divide(r, axis=axis), el.divide(er, axis=axis))
            assert_eq(l.truediv(r, axis=axis), el.truediv(er, axis=axis))
            assert_eq(l.floordiv(r, axis=axis), el.floordiv(er, axis=axis))
            assert_eq(l.mod(r, axis=axis), el.mod(er, axis=axis))
            assert_eq(l.pow(r, axis=axis), el.pow(er, axis=axis))
            assert_eq(l.rdiv(r, axis=axis), el.rdiv(er, axis=axis))
            assert_eq(l.rtruediv(r, axis=axis), el.rtruediv(er, axis=axis))
            assert_eq(l.rpow(r, axis=axis), el.rpow(er, axis=axis))
            assert_eq(l.rmod(r, axis=axis), el.rmod(er, axis=axis))
            assert_eq(l.radd(r, axis=axis), el.radd(er, axis=axis))
            assert_eq(l.rsub(r, axis=axis), el.rsub(er, axis=axis))
            assert_eq(l.rmul(r, axis=axis), el.rmul(er, axis=axis))