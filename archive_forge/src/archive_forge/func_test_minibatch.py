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
def test_minibatch():
    fix_random_seed(0)
    ops = get_current_ops()
    items = [1, 2, 3, 4, 5, 6]
    batches = ops.minibatch(3, items)
    assert list(batches) == [[1, 2, 3], [4, 5, 6]]
    batches = ops.minibatch((i for i in (3, 2, 1)), items)
    assert list(batches) == [[1, 2, 3], [4, 5], [6]]
    batches = list(ops.minibatch(3, numpy.asarray(items)))
    assert isinstance(batches[0], numpy.ndarray)
    assert numpy.array_equal(batches[0], numpy.asarray([1, 2, 3]))
    assert numpy.array_equal(batches[1], numpy.asarray([4, 5, 6]))
    batches = list(ops.minibatch((i for i in (3, 2, 1)), items, shuffle=True))
    assert batches != [[1, 2, 3], [4, 5], [6]]
    assert len(batches[0]) == 3
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1
    with pytest.raises(ValueError):
        ops.minibatch(10, (i for i in range(100)))
    with pytest.raises(ValueError):
        ops.minibatch(10, True)