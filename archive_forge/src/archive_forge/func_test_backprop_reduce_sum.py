import inspect
import platform
from typing import Tuple, cast
import numpy
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, integers
from numpy.testing import assert_allclose
from packaging.version import Version
from thinc.api import (
from thinc.backends._custom_kernels import KERNELS, KERNELS_LIST, compile_mmh
from thinc.compat import has_cupy_gpu, has_torch, torch_version
from thinc.types import Floats2d
from thinc.util import torch2xp, xp2torch
from .. import strategies
from ..strategies import arrays_BI, ndarrays_of_shape
@pytest.mark.parametrize('ops', ALL_OPS)
@pytest.mark.parametrize('dtype', FLOAT_TYPES)
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_backprop_reduce_sum(ops, dtype, X):
    X = ops.asarray(X, dtype=dtype)
    if ops.xp.abs(X).max() >= 5:
        return None
    lengths = ops.asarray([3] * len(X), dtype='i')
    out = ops.backprop_reduce_sum(X, lengths)
    assert out.dtype == dtype
    assert out.shape == (sum(lengths), X.shape[1])
    start = 0
    for i, length in enumerate(lengths):
        ops.xp.testing.assert_allclose(out[start:start + length].sum(axis=0), X[i] * length, rtol=0.01, atol=0.01)
        start += length