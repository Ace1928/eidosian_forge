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
@pytest.mark.parametrize('npfunc,daskfunc', [(np.sort, da.topk), (np.argsort, da.argtopk)])
@pytest.mark.parametrize('split_every', [None, 2, 4, 8])
def test_topk_argtopk1(npfunc, daskfunc, split_every):
    k = 5
    rng = np.random.default_rng()
    npa = rng.random(800)
    npb = rng.random((10, 20, 30))
    a = da.from_array(npa, chunks=((120, 80, 100, 200, 300),))
    b = da.from_array(npb, chunks=(4, 8, 8))
    assert_eq(npfunc(npa)[-k:][::-1], daskfunc(a, k, split_every=split_every))
    assert_eq(npfunc(npa)[:k], daskfunc(a, -k, split_every=split_every))
    assert_eq(npfunc(npb, axis=0)[-k:, :, :][::-1, :, :], daskfunc(b, k, axis=0, split_every=split_every))
    assert_eq(npfunc(npb, axis=1)[:, -k:, :][:, ::-1, :], daskfunc(b, k, axis=1, split_every=split_every))
    assert_eq(npfunc(npb, axis=-1)[:, :, -k:][:, :, ::-1], daskfunc(b, k, axis=-1, split_every=split_every))
    with pytest.raises(ValueError):
        daskfunc(b, k, axis=3, split_every=split_every)
    assert_eq(npfunc(npb, axis=0)[:k, :, :], daskfunc(b, -k, axis=0, split_every=split_every))
    assert_eq(npfunc(npb, axis=1)[:, :k, :], daskfunc(b, -k, axis=1, split_every=split_every))
    assert_eq(npfunc(npb, axis=-1)[:, :, :k], daskfunc(b, -k, axis=-1, split_every=split_every))
    with pytest.raises(ValueError):
        daskfunc(b, -k, axis=3, split_every=split_every)