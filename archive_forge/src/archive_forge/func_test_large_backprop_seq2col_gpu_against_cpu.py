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
def test_large_backprop_seq2col_gpu_against_cpu(nW):
    cupy_ops = CupyOps()
    numpy_ops = NumpyOps()
    batch_size = 128 * 128 * 2
    nF = 2 * nW + 1
    d_cols = numpy_ops.xp.random.randn(batch_size * nF).astype('float32').reshape(-1, nF)
    d_cols_gpu = cupy_ops.asarray2f(d_cols)
    lengths = numpy_ops.asarray1i([1, 4, 2, 1] * (batch_size // 8))
    lengths_gpu = cupy_ops.asarray1i(lengths)
    d_seqs = numpy_ops.backprop_seq2col(d_cols, nW=nW, lengths=lengths)
    d_seqs_gpu = cupy_ops.backprop_seq2col(d_cols_gpu, nW=nW, lengths=lengths_gpu)
    assert_allclose(d_seqs, d_seqs_gpu.get())