from __future__ import annotations
import pytest
import operator
import numpy as np
import dask.array as da
from dask.array.chunk import coarsen, getitem, keepdims_wrapper
def test_keepdims_wrapper_two_axes():

    def summer(a, axis=None):
        return a.sum(axis=axis)
    summer_wrapped = keepdims_wrapper(summer)
    assert summer_wrapped != summer
    a = np.arange(24).reshape(1, 2, 3, 4)
    r = summer(a, axis=(1, 3))
    rw = summer_wrapped(a, axis=(1, 3), keepdims=True)
    rwf = summer_wrapped(a, axis=(1, 3), keepdims=False)
    assert r.ndim == 2
    assert r.shape == (1, 3)
    assert (r == np.array([[60, 92, 124]])).all()
    assert rw.ndim == 4
    assert rw.shape == (1, 1, 3, 1)
    assert (rw == np.array([[[[60], [92], [124]]]])).all()
    assert rwf.ndim == 2
    assert rwf.shape == (1, 3)
    assert (rwf == np.array([[60, 92, 124]])).all()