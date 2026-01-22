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
def test_reduce_max_sm(ops, dtype):
    X = ops.xp.zeros((6, 3), dtype=dtype)
    X += ops.xp.random.uniform(-1, 1, X.shape)
    lengths = ops.xp.array([2, 2, 2], dtype='i')
    maxes, which = ops.reduce_max(X, lengths)
    assert maxes.dtype == dtype
    assert ops.xp.all(which >= 0)
    assert ops.xp.all(which < X.shape[0])
    start = 0
    for i, length in enumerate(lengths):
        truth = X[start:start + length].max(axis=0)
        ops.xp.testing.assert_allclose(maxes[i], truth)
        start += length