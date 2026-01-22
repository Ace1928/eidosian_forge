from __future__ import annotations
import os
import warnings
from contextlib import nullcontext as does_not_warn
from itertools import permutations, zip_longest
import pytest
import itertools
import dask.array as da
import dask.config as config
from dask.array.numpy_compat import NUMPY_GE_122, ComplexWarning
from dask.array.utils import assert_eq, same_keys
from dask.core import get_deps
def reduction_2d_test(da_func, darr, np_func, narr, use_dtype=True, split_every=True):
    assert_eq(da_func(darr), np_func(narr))
    assert_eq(da_func(darr, keepdims=True), np_func(narr, keepdims=True))
    assert_eq(da_func(darr, axis=()), np_func(narr, axis=()))
    assert_eq(da_func(darr, axis=0), np_func(narr, axis=0))
    assert_eq(da_func(darr, axis=1), np_func(narr, axis=1))
    assert_eq(da_func(darr, axis=-1), np_func(narr, axis=-1))
    assert_eq(da_func(darr, axis=-2), np_func(narr, axis=-2))
    assert_eq(da_func(darr, axis=1, keepdims=True), np_func(narr, axis=1, keepdims=True))
    assert_eq(da_func(darr, axis=(), keepdims=True), np_func(narr, axis=(), keepdims=True))
    assert_eq(da_func(darr, axis=(1, 0)), np_func(narr, axis=(1, 0)))
    assert same_keys(da_func(darr, axis=()), da_func(darr, axis=()))
    assert same_keys(da_func(darr, axis=1), da_func(darr, axis=1))
    assert same_keys(da_func(darr, axis=(1, 0)), da_func(darr, axis=(1, 0)))
    if use_dtype:
        with pytest.warns(ComplexWarning) if np.iscomplexobj(narr) else does_not_warn():
            assert_eq(da_func(darr, dtype='f8'), np_func(narr, dtype='f8'))
            assert_eq(da_func(darr, dtype='i8'), np_func(narr, dtype='i8'))
    if split_every:
        a1 = da_func(darr, split_every=4)
        a2 = da_func(darr, split_every={0: 2, 1: 2})
        assert same_keys(a1, a2)
        assert_eq(a1, np_func(narr))
        assert_eq(a2, np_func(narr))
        assert_eq(da_func(darr, keepdims=True, split_every=4), np_func(narr, keepdims=True))
        assert_eq(da_func(darr, axis=(), split_every=2), np_func(narr, axis=()))
        assert_eq(da_func(darr, axis=0, split_every=2), np_func(narr, axis=0))
        assert_eq(da_func(darr, axis=(), keepdims=True, split_every=2), np_func(narr, axis=(), keepdims=True))
        assert_eq(da_func(darr, axis=0, keepdims=True, split_every=2), np_func(narr, axis=0, keepdims=True))
        assert_eq(da_func(darr, axis=1, split_every=2), np_func(narr, axis=1))
        assert_eq(da_func(darr, axis=1, keepdims=True, split_every=2), np_func(narr, axis=1, keepdims=True))