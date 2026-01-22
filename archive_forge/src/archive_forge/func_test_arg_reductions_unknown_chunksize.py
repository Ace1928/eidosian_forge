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
@pytest.mark.parametrize('func', ['argmax', 'nanargmax'])
def test_arg_reductions_unknown_chunksize(func):
    x = da.arange(10, chunks=5)
    x = x[x > 1]
    with pytest.raises(ValueError) as info:
        getattr(da, func)(x)
    assert 'unknown chunksize' in str(info.value)