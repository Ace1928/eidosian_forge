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
def test_ngrams():
    ops = get_current_ops()
    arr1 = numpy.asarray([1, 2, 3, 4, 5], dtype=numpy.uint64)
    for n in range(1, 10):
        assert len(ops.ngrams(n, arr1)) == max(0, arr1.shape[0] - (n - 1))
    assert len(ops.ngrams(-1, arr1)) == 0
    assert len(ops.ngrams(arr1.shape[0] + 1, arr1)) == 0