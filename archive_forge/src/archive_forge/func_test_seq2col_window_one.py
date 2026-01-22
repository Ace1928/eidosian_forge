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
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(X=strategies.arrays_BI())
def test_seq2col_window_one(ops, X):
    X = ops.asarray(X)
    base_ops = Ops()
    base_ops.xp = ops.xp
    baseX = base_ops.alloc(X.shape) + X
    target = base_ops.seq2col(base_ops.asarray(baseX), nW=1)
    predicted = ops.seq2col(X, nW=1)
    ops.xp.testing.assert_allclose(target, predicted, atol=0.001, rtol=0.001)