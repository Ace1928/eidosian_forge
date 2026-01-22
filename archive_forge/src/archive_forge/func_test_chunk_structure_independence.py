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
@pytest.mark.parametrize('chunks', list(permutations(((2, 1) * 8, (3,) * 8, (6,) * 4))))
@pytest.mark.parametrize('split_every', [2, 4])
@pytest.mark.parametrize('axes', list(permutations((0, 1, 2), 2)) + list(permutations((0, 1, 2))))
def test_chunk_structure_independence(axes, split_every, chunks):
    shape = tuple((np.sum(s) for s in chunks))
    np_array = np.arange(np.prod(shape)).reshape(*shape)
    x = da.from_array(np_array, chunks=chunks)
    reduced_x = da.reduction(x, lambda x, axis, keepdims: x, lambda x, axis, keepdims: x, keepdims=True, axis=axes, split_every=split_every, dtype=x.dtype, meta=x._meta)
    assert_eq(reduced_x, np_array, check_chunks=False, check_shape=False)