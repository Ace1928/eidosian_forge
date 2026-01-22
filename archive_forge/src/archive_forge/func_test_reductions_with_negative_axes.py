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
def test_reductions_with_negative_axes():
    x = np.random.default_rng().random((4, 4, 4))
    a = da.from_array(x, chunks=2)
    assert_eq(a.argmin(axis=-1), x.argmin(axis=-1))
    assert_eq(a.argmin(axis=-1, split_every=2), x.argmin(axis=-1))
    assert_eq(a.sum(axis=-1), x.sum(axis=-1))
    assert_eq(a.sum(axis=(0, -1)), x.sum(axis=(0, -1)))