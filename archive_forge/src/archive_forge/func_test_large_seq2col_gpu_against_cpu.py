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
@pytest.mark.skipif(not has_cupy_gpu, reason='needs GPU/CuPy')
@pytest.mark.parametrize('nW', [1, 2])
def test_large_seq2col_gpu_against_cpu(nW):
    cupy_ops = CupyOps()
    numpy_ops = NumpyOps()
    batch_size = 128 * 128 * 2
    X = numpy_ops.xp.random.randn(batch_size * 2).astype('float32').reshape(-1, 2)
    X_gpu = cupy_ops.asarray2f(X)
    lengths = numpy_ops.asarray1i([1, 4, 2, 1] * (batch_size // 8))
    lengths_gpu = cupy_ops.asarray1i(lengths)
    cols = numpy_ops.seq2col(X, nW=nW, lengths=lengths)
    cols_gpu = cupy_ops.seq2col(X_gpu, nW=nW, lengths=lengths_gpu)
    assert_allclose(cols, cols_gpu.get())