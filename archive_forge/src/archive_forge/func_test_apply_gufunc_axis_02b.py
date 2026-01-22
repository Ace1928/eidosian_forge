from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_apply_gufunc_axis_02b():

    def myfilter(x, cn=10, axis=-1):
        y = np.fft.fft(x, axis=axis)
        y[cn:-cn] = 0
        nx = np.fft.ifft(y, axis=axis)
        return np.real(nx)
    a = np.random.default_rng().standard_normal((3, 6, 4))
    da_ = da.from_array(a, chunks=2)
    m = myfilter(a, axis=1)
    dm = apply_gufunc(myfilter, '(i)->(i)', da_, axis=1, allow_rechunk=True)
    assert_eq(m, dm)