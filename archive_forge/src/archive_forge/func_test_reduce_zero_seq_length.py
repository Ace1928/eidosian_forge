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
@pytest.mark.parametrize('reduction_raises', REDUCE_ZERO_LENGTH_RAISES)
def test_reduce_zero_seq_length(ops, dtype, reduction_raises):
    reduction_str, raises = reduction_raises
    reduction = getattr(ops, reduction_str)
    X = ops.asarray2f([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    lengths = ops.asarray1i([3, 0, 2])
    if raises:
        with pytest.raises(ValueError):
            reduction(X, lengths)
    else:
        ops.xp.testing.assert_allclose(reduction(X, lengths)[1], [0.0, 0.0])