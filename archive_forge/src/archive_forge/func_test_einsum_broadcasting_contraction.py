from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
def test_einsum_broadcasting_contraction():
    rng = np.random.default_rng()
    a = rng.random((1, 5, 4))
    b = rng.random((4, 6))
    c = rng.random((5, 6))
    d = rng.random(10)
    d_a = da.from_array(a, chunks=(1, (2, 3), (2, 2)))
    d_b = da.from_array(b, chunks=((2, 2), (4, 2)))
    d_c = da.from_array(c, chunks=((2, 3), (4, 2)))
    d_d = da.from_array(d, chunks=(7, 3))
    np_res = np.einsum('ijk,kl,jl', a, b, c)
    da_res = da.einsum('ijk,kl,jl', d_a, d_b, d_c)
    assert_eq(np_res, da_res)
    mul_res = da_res * d
    np_res = np.einsum('ijk,kl,jl,i->i', a, b, c, d)
    da_res = da.einsum('ijk,kl,jl,i->i', d_a, d_b, d_c, d_d)
    assert_eq(np_res, da_res)
    assert_eq(np_res, mul_res)