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
@given(x=strategies.floats(min_value=-10, max_value=10))
def test_clipped_linear(ops, x):
    x_thinc = ops.xp.asarray([x])
    assert ops.xp.isclose(ops.clipped_linear(x_thinc, max_val=6.0), ops.relu_k(x_thinc))
    assert ops.xp.isclose(ops.backprop_clipped_linear(ops.asarray1f([1.0]), x_thinc, max_val=6.0), ops.backprop_relu_k(ops.asarray1f([1.0]), x_thinc))
    assert ops.xp.isclose(ops.clipped_linear(x_thinc, slope=0.2, offset=0.5), ops.hard_sigmoid(x_thinc))
    assert ops.xp.isclose(ops.backprop_clipped_linear(ops.asarray1f([1.0]), x_thinc, slope=0.2, offset=0.5), ops.backprop_hard_sigmoid(ops.asarray1f([1.0]), x_thinc))