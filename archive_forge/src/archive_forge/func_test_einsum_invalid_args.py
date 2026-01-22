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
def test_einsum_invalid_args():
    _, da_inputs = _numpy_and_dask_inputs('a')
    with pytest.raises(TypeError):
        da.einsum('a', *da_inputs, foo=1, bar=2)