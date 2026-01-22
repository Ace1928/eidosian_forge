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
@given(X=strategies.arrays_BOP())
def test_maxout(ops, dtype, X):
    X = ops.asarray(X, dtype=dtype)
    expected_best = X.max(axis=-1).astype(dtype)
    predicted_best, which = ops.maxout(X)
    assert predicted_best.dtype == dtype
    ops.xp.testing.assert_allclose(expected_best, predicted_best, rtol=0.001, atol=0.001)
    ops.xp.testing.assert_allclose(ops.xp.take_along_axis(X, ops.xp.expand_dims(which, -1), axis=-1), ops.xp.expand_dims(expected_best, -1), atol=1e-10)