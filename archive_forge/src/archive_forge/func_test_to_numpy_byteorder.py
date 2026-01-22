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
@pytest.mark.parametrize('byte_order', ('>', '<', '=', '|'))
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(x=strategies.floats(min_value=-10, max_value=10))
def test_to_numpy_byteorder(ops, byte_order, x):
    x = ops.xp.asarray([x])
    y = ops.to_numpy(x, byte_order=byte_order)
    assert numpy.array_equal(ops.to_numpy(x), ops.to_numpy(y))
    if byte_order in ('>', '<'):
        assert y.dtype.newbyteorder('S').newbyteorder('S').byteorder == byte_order
    else:
        assert x.dtype.byteorder == y.dtype.byteorder