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
@pytest.mark.skipif(not has_torch, reason='needs PyTorch')
@pytest.mark.parametrize('ops', ALL_OPS)
@pytest.mark.parametrize('temperature', [0.5, 1.0, 2.0])
def test_softmax_temperature(ops, temperature):
    X = ops.xp.arange(-10, 10, 0.2, dtype='f').reshape(10, 10)
    dY = ops.xp.eye(10, dtype='f')
    Y = ops.softmax(X, temperature=temperature)
    dX = ops.backprop_softmax(Y, dY, temperature=temperature)
    Yt, dXt = torch_softmax_with_temperature(X, dY, temperature)
    ops.xp.testing.assert_allclose(Y, Yt, atol=1e-06)
    ops.xp.testing.assert_allclose(dX, dXt, atol=1e-06)