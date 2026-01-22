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
@pytest.mark.parametrize('ops', XP_OPS)
@pytest.mark.parametrize('dtype', FLOAT_TYPES)
def test_seq2col_lengths_all_zero(ops, dtype):
    ops.xp.testing.assert_allclose(ops.alloc((0, 0), dtype=dtype), ops.seq2col(ops.alloc((0, 0), dtype=dtype), 1, lengths=ops.xp.zeros((0,), dtype='int32')))
    ops.xp.testing.assert_allclose(ops.alloc((0, 0), dtype=dtype), ops.backprop_seq2col(ops.alloc((0, 0), dtype=dtype), 1, lengths=ops.xp.zeros((0,), dtype='int32')))
    ops.xp.testing.assert_allclose(ops.alloc((0, 0), dtype=dtype), ops.seq2col(ops.alloc((0, 0), dtype=dtype), 1, lengths=ops.asarray1i([0])))
    ops.xp.testing.assert_allclose(ops.alloc((0, 0), dtype=dtype), ops.backprop_seq2col(ops.alloc((0, 0), dtype=dtype), 1, lengths=ops.asarray1i([0])))
    ops.xp.testing.assert_allclose(ops.alloc((0, 0), dtype=dtype), ops.seq2col(ops.alloc((0, 0), dtype=dtype), 1, lengths=ops.asarray1i([0, 0])))
    ops.xp.testing.assert_allclose(ops.alloc((0, 0), dtype=dtype), ops.backprop_seq2col(ops.alloc((0, 0), dtype=dtype), 1, lengths=ops.asarray1i([0, 0])))