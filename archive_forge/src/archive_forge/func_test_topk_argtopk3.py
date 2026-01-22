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
def test_topk_argtopk3():
    a = da.random.default_rng().random((10, 20, 30), chunks=(4, 8, 8))
    assert_eq(a.topk(5, axis=1, split_every=2), da.topk(a, 5, axis=1, split_every=2))
    assert_eq(a.argtopk(5, axis=1, split_every=2), da.argtopk(a, 5, axis=1, split_every=2))