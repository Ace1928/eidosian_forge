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
def test_backprop_reduce_max(ops, dtype):
    dX = ops.backprop_reduce_max(ops.xp.arange(1, 7, dtype=dtype).reshape(2, 3), ops.xp.array([[2, 1, 0], [1, 0, 1]]).astype('int32'), ops.xp.array([3, 2], dtype='int32'))
    assert dX.dtype == dtype
    ops.xp.testing.assert_allclose(dX, [[0.0, 0.0, 3.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [4.0, 0.0, 6.0]])
    with pytest.raises(IndexError):
        ops.backprop_reduce_max(ops.xp.arange(1, 7, dtype='f').reshape(2, 3), ops.xp.array([[2, 3, 0], [1, 0, 1]]).astype('int32'), ops.xp.array([3, 2], dtype='int32'))
    with pytest.raises(ValueError):
        ops.backprop_reduce_max(ops.xp.arange(1, 7, dtype=dtype).reshape(2, 3), ops.xp.array([[2, 1, 0], [1, 0, 1]]).astype('int32'), ops.xp.array([-3, 2], dtype='int32'))
    with pytest.raises(ValueError):
        ops.backprop_reduce_max(ops.xp.arange(1, 7, dtype=dtype).reshape(2, 3), ops.xp.array([[2, 1, 0], [1, 0, 1], [1, 0, 1]]).astype('int32'), ops.xp.array([3, 0, 2], dtype='int32'))