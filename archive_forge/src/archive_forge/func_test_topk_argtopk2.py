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
@pytest.mark.parametrize('split_every', [None, 2, 3, 4])
@pytest.mark.parametrize('chunksize', [1, 2, 3, 4, 5, 10])
def test_topk_argtopk2(npfunc, daskfunc, split_every, chunksize):
    """Fine test use cases when k is larger than chunk size"""
    npa = np.random.default_rng().random((10,))
    a = da.from_array(npa, chunks=chunksize)
    k = 5
    assert_eq(npfunc(npa)[-k:][::-1], daskfunc(a, k, split_every=split_every))
    assert_eq(npfunc(npa)[:k], daskfunc(a, -k, split_every=split_every))